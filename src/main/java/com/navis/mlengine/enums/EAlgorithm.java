package com.navis.mlengine.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum EAlgorithm {
    UNKNOWN(0, "Invalid Algorithm"),
    XGBOOST(1, "Extreme Gradient boost algorithm"),
    NN(2, "Neural Networks Algorithm");

    private final int id;
    private final String description;

    @JsonCreator
    public static Boolean isValidAlgorithm(final String chosenAlgo) {
        for (final EAlgorithm a : EAlgorithm.values()) {
            if (a.name().equals(chosenAlgo) == true) {
                return true;
            }
        }
        return false;
    }

}
