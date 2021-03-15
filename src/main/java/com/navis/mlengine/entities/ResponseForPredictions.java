package com.navis.mlengine.entities;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
public class ResponseForPredictions {
    public List<Float> predictions;
    public Double rmse;
    public Double mape;
    public Double bias;
    public Double accuracyPercentage;
}
