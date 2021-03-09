package com.navis.mlengine.mlhelpers.encoders;

import com.navis.mlengine.enums.EFeatureType;

import java.util.ArrayList;

public abstract class BaseEncoder {
    protected ArrayList<ArrayList<String>> featureMatrix;
    protected ArrayList<EFeatureType> featureTypes;
    ArrayList<String> classValues;
    EFeatureType classType;
}
