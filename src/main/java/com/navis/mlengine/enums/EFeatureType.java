package com.navis.mlengine.enums;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum EFeatureType {
    UNKNOWN(0, "Invalid feature datatype"),
    CATEGORY(1, "Catogorical feature"),
    NUMBER(2, "Numerical feature(includes integers, floats and doubles)"),
    BOOLEAN(3,"Boolean feature"),
    STRING(4,"String feature");

    private final int id;
    private final String description;
}
