package com.navis.mlengine.mlhelpers.encoders;

import com.navis.mlengine.enums.EFeatureType;
import javafx.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;

public class LabelEncoder extends BaseEncoder {
    //encodingDetails is for reconstruction - Intgeger in the pair corresponds to the column(0 indexed)
    //and hashmap contains value->encoding. List of the whole thing becuase we can have many columns encoded
    protected HashMap<Integer, HashMap<String, Integer>> encodingDetails = new HashMap<>();



    protected ArrayList<ArrayList<String>> encodedFeatureMatrix;

    public LabelEncoder(ArrayList<ArrayList<String>> inFeatureMatrix, ArrayList<EFeatureType> inFeatureTypes) {
        super.featureMatrix = inFeatureMatrix;
        super.featureTypes = inFeatureTypes;
    }

    public Pair<ArrayList<ArrayList<String>>, HashMap<String, Integer>> EncodeFeatureMatrixForModelCreation(ArrayList<ArrayList<String>> inMatrix, Integer indexToEncode) { //0-indexed
        if(featureMatrix == null)
            return null;

        ArrayList<ArrayList<String>> encodedData = new ArrayList<>();
        HashMap<String, Integer> encoding = new HashMap<>();
        Integer maxEncode = 1;

        for(ArrayList<String> dataRow : inMatrix) {
            ArrayList<String> encodedRow = new ArrayList<>();
            for(int col = 0; col<dataRow.size(); col++) {
                if(col == indexToEncode) {
                    Integer value = encoding.get(dataRow.get(col));
                    if(value != null) {
                        encodedRow.add(value.toString());       //Put is as a string later will be converted before passing to algorithm as the last stage
                    } else {
                        encoding.put(dataRow.get(col), maxEncode);
                        encodedRow.add(maxEncode.toString());
                        maxEncode++;
                    }
                } else {
                    encodedRow.add(dataRow.get(col));
                }
            }

            encodedData.add(encodedRow);
        }

        inMatrix = encodedData;

        //append fill encoding details of this call
        encodingDetails.put(indexToEncode, encoding);

        return new Pair(encodedData, encoding);
    }

    /*public Pair<ArrayList<ArrayList<String>>, HashMap<String, Integer>> EncodeForPrediction() {
    }*/
}
