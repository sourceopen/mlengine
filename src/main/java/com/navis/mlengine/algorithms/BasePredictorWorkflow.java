package com.navis.mlengine.algorithms;

import com.navis.mlengine.configuration.GenericAlgorithmConfigurationBundle;
import com.navis.mlengine.entities.MLModel;
import com.navis.mlengine.entities.ResponseForTraining;
import com.navis.mlengine.enums.EFeatureType;
import org.javatuples.Quartet;
import org.javers.common.collections.Pair;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public abstract class BasePredictorWorkflow {
    protected ArrayList<ArrayList<String>> matrixForPrediction;
    protected ArrayList<ArrayList<String>> predictions;

    public abstract Pair<ResponseForTraining, MLModel> buildAndSaveModel(GenericAlgorithmConfigurationBundle mlBundle);
    public abstract List<Float> predictFromModelForRegression(MLModel model, GenericAlgorithmConfigurationBundle bundle);
    public abstract List<Float> predictFromModelForBinaryClassification(MLModel model, GenericAlgorithmConfigurationBundle bundle);


    protected ArrayList<ArrayList<String>> featureMatrix;   //Corresponds to the Xs in the below diagram
    protected ArrayList<EFeatureType> featureTypes;        //Corresponds to the Ys in the below diagram
    protected ArrayList<String> classValues;         //Corresponds to the As in the below diagram
    protected EFeatureType classType;        //Corresponds to the Z in the below diagram

/*
rawDataTypes is Y,Y,Y,Z(from mlconfig.json)
rawData is,
      ________________________
     |__Y__|__Y__|__Y__|__Z__|
     |  X  |  X  |  X  |  A  |
     |  X  |  X  |  X  |  A  |
     |  X  |  X  |  X  |  A  |
     |  X  |  X  |  X  |  A  |
     |  X  |  X  |  X  |  A  |
     |  X  |  X  |  X  |  A  |
     |__X__|__X__|__X__|__A__|

seperateFeatureAndClass() method here splits all this into the different
component mentioned above and fills these class data members from the rawData
 */

    protected ArrayList<ArrayList<String>> dataForPrediction;
    protected ArrayList<ArrayList<String>> curatedDataForPrediction;

    public Boolean elementaryCheckForPrediction(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        if(mlAlgorithmConfigurationBundle.getRawDataForPrediction().size() == 0)
            return false;

        //Check if the size of the data types are the same as the size of the data - label
        if (mlAlgorithmConfigurationBundle.getFeatureTypesWithoutClassForPrediction().size() != mlAlgorithmConfigurationBundle.getRawDataForPrediction().get(0).size()) {
            return false;
        }

        //Check if all rows have the same size
        int sizeOfEachRow = mlAlgorithmConfigurationBundle.getFeatureTypesWithoutClassForPrediction().size();
        for(ArrayList<String> r : mlAlgorithmConfigurationBundle.getRawDataForPrediction()) {
            if(r.size() != sizeOfEachRow)
                return false;
        }

        return true;
    }

    public Boolean elementaryCheck(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        if(mlAlgorithmConfigurationBundle.getRawTrainingData().size() == 0) //There is atleast one row to train on
            return false;

        //Check if the size of the data types are the same as the size of the data
        if (mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().size() != mlAlgorithmConfigurationBundle.getRawTrainingData().get(0).size()) {
            return false;
        }

        //Check if all rows have the same size
        int sizeOfEachRow = mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().size();
        for(ArrayList<String> r : mlAlgorithmConfigurationBundle.getRawTrainingData()) {
            if(r.size() != sizeOfEachRow)
                return false;
        }

        return true;
    }

    public Boolean areCatogoricalFeaturesAvailable(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        for(EFeatureType fT : mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass()) {
            if(fT == EFeatureType.CATEGORY)
                return true;
        }

        return false;
    }

    public Quartet<ArrayList<ArrayList<String>>, ArrayList<EFeatureType>, ArrayList<String>, EFeatureType> seperateFeatureAndClass(
                                                                        GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {   //class should be the last column
        ArrayList<ArrayList<String>> justFeatureMatrix = new ArrayList<>();
        ArrayList<EFeatureType> featureTypesOfMatrix = new ArrayList<>();
        ArrayList<String> classes = new ArrayList<>();
        EFeatureType typeOfClass;

        for(ArrayList<String> r : mlAlgorithmConfigurationBundle.getRawTrainingData()) {
            ArrayList<String> featureMatrixRow = new ArrayList<>();
            for(int col = 0;col < (r.size()-1); col++) {
                featureMatrixRow.add(r.get(col));
            }
            justFeatureMatrix.add(featureMatrixRow);
            classes.add(r.get(mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().size()-1));     //Since features size and matrix col size is same, this points to the last col - the class
        }

        for(int col = 0;col < (mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().size()-1); col++) {
            featureTypesOfMatrix.add(mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().get(col));
        }

        typeOfClass = EFeatureType.valueOf(mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().get(mlAlgorithmConfigurationBundle.getFeatureTypesIncludingClass().size()-1).name());

        featureMatrix = justFeatureMatrix;
        featureTypes = featureTypesOfMatrix;
        classValues = classes;
        classType = typeOfClass;

        return new Quartet<>(justFeatureMatrix, featureTypesOfMatrix, classes, typeOfClass);
    }

    public void collectTestDataFromInput(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        ArrayList<ArrayList<String>> featureMatrix = new ArrayList<>();
        ArrayList<EFeatureType> featureTypesOfMatrix = new ArrayList<>();

        for(ArrayList<String> r : mlAlgorithmConfigurationBundle.getRawDataForPrediction()) {
            ArrayList<String> featuresRow = new ArrayList<>();
            for(int col = 0;col < r.size(); col++) {
                featuresRow.add(r.get(col));
            }
            featureMatrix.add(featuresRow);
        }

        for(int col = 0;col < (mlAlgorithmConfigurationBundle.getFeatureTypesWithoutClassForPrediction().size()-1); col++) {
            featureTypesOfMatrix.add(mlAlgorithmConfigurationBundle.getFeatureTypesWithoutClassForPrediction().get(col));
        }

        dataForPrediction = featureMatrix;
        featureTypes = featureTypesOfMatrix;

        return;
    }


}
