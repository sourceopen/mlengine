package com.navis.mlengine.enums;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum EEncodingType {
    LABEL(0, "Label encoding"),
    OHE(1, "One hot encoding");

    private final int id;
    private final String description;
}
