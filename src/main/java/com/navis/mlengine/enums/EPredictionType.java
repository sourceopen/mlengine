package com.navis.mlengine.enums;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum EPredictionType {
    UNKNOWN(0, "Unsupported Prediction type"),
    MULTICLASS_CLASSIFICATION(1, "Classify into one of multiple categories(>2) based on previous data"),
    BINARY_CLASSIFICATION(2, "Classify into one of two categories(eg yes/no) based on previous data"),
    REGRESSION(3,"Predict a continuous outcome based on the features and previous data");

    private final int id;
    private final String description;
}
