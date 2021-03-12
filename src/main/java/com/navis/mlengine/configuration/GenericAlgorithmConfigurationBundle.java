package com.navis.mlengine.configuration;

import com.navis.mlengine.enums.EFeatureType;
import com.navis.mlengine.enums.EPredictionType;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.ArrayList;


@Setter
@Getter
@AllArgsConstructor
@NoArgsConstructor
public class GenericAlgorithmConfigurationBundle {
    protected String consumerId;

    //All raw data lives here
    protected ArrayList<ArrayList<String>> rawTrainingData;    //Length n, n-1 is data and n is the label of the data

    protected ArrayList<ArrayList<String>> rawDataForPrediction;

    protected ArrayList<EFeatureType> featureTypesIncludingClass;    //1-1 correspondence with trainingData/testData
    protected ArrayList<EFeatureType> featureTypesWithoutClassForPrediction;    //1-1 correspondence with trainingData/testData
    protected ArrayList<String> featureNames;
    protected EPredictionType predictionType;
}
