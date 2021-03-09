package com.navis.mlengine.configuration;

import com.navis.mlengine.enums.EFeatureType;
import com.navis.mlengine.enums.EPredictionType;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.ArrayList;

@Setter
@AllArgsConstructor
@Getter
@NoArgsConstructor
public class XGBoostConfigurationBundle extends GenericAlgorithmConfigurationBundle {
    private double eta = 0.3;   //step size/learning rate
    private int rounds = 100;
    private double alpha = 0;   //L1 regularization term
    private double lambda = 1;  //L2 regularization term
    private double gamma = 0;
    private int verbosity = 1;
    private int max_depth = 7;
    private int max_child_wt = 3;
    private int min_child_wt = 3;
    private int silent = 0;
    private double subsample = 0.8;
    private String objective;
    private String eval_metric;
    private double max_delta_step;
    private String booster;
    private int n_estimators;

    public XGBoostConfigurationBundle(String consumerId, ArrayList<ArrayList<String>> trainingData, ArrayList<EFeatureType> featureTypes,
                                      ArrayList<String> inFeatureNames, EPredictionType predictionType) {
        super(consumerId, trainingData, featureTypes, inFeatureNames, predictionType);
    }
}
