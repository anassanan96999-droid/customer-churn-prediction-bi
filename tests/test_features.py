import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_preprocessing import load_raw_csv, prepare_features
from src.feature_engineering import build_preprocessor, output_feature_groups


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    raw = load_raw_csv()
    blank = raw["TotalCharges"].str.strip() == ""
    # the 11 blank-TotalCharges customers (tenure 0) plus a random slice of the rest
    return pd.concat([raw[blank], raw[~blank].sample(200, random_state=1)])


def test_duplicate_customer_rows_are_collapsed(raw_sample):
    doubled = pd.concat([raw_sample, raw_sample.head(5)])
    assert len(prepare_features(doubled)) == len(raw_sample)


def test_blank_total_charges_become_zero_not_nan(raw_sample):
    feats = prepare_features(raw_sample)
    new = feats[feats["tenure"] == 0]
    assert len(new) == 11
    assert (new["total_charges"] == 0).all()
    # an unbilled customer's average spend is their current bill, trend is neutral
    assert np.allclose(new["avg_monthly_spend"], new["monthly_charges"])
    assert (new["spend_trend_ratio"] == 1.0).all()


def test_model_features_have_no_missing_values(raw_sample):
    feats = prepare_features(raw_sample)
    assert feats[config.MODEL_FEATURES].isna().sum().sum() == 0


def test_no_internet_service_collapses_to_no(raw_sample):
    feats = prepare_features(raw_sample)
    for col in ("online_security", "tech_support", "streaming_tv"):
        assert set(feats[col].unique()) <= {"Yes", "No"}


def test_upload_without_churn_column_is_accepted(raw_sample):
    feats = prepare_features(raw_sample.drop(columns="Churn"))
    assert config.TARGET not in feats.columns
    assert len(feats) == len(raw_sample)


def test_missing_columns_are_reported(raw_sample):
    with pytest.raises(ValueError, match="Contract"):
        prepare_features(raw_sample.drop(columns="Contract"))


def test_every_encoded_column_maps_back_to_a_feature(raw_sample):
    feats = prepare_features(raw_sample)
    pre = build_preprocessor().fit(feats[config.MODEL_FEATURES])
    groups = output_feature_groups(pre)
    assert len(groups) == pre.transform(feats[config.MODEL_FEATURES].head(3)).shape[1]
    assert set(groups) == set(config.MODEL_FEATURES)
