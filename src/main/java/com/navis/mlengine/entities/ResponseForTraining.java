package com.navis.mlengine.entities;

import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class ResponseForTraining {
    List<Float> actuals;
    List<Float> predictions;
    Double rmse;
    Double mape;
}
