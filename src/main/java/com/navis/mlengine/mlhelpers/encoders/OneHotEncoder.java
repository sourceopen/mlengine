package com.navis.mlengine.mlhelpers.encoders;

import com.navis.mlengine.enums.EFeatureType;
import javafx.util.Pair;

import java.util.*;

public class OneHotEncoder extends BaseEncoder {
    //encodingDetails is for reconstruction - Intgeger in the pair corresponds to the column(0 indexed)
    //and hashmap contains value->encoding. List of the whole thing becuase we can have many columns encoded
    protected HashMap<Integer, HashMap<String, ArrayList<Integer>>> encodingDetails = new HashMap<>();

    protected ArrayList<ArrayList<String>> encodedFeatureMatrix;

    public OneHotEncoder(ArrayList<ArrayList<String>> inFeatureMatrix, ArrayList<EFeatureType> inFeatureTypes) {
        super.featureMatrix = inFeatureMatrix;
        super.featureTypes = inFeatureTypes;
    }

    public Pair<ArrayList<ArrayList<String>>, HashMap<String, List<Integer>>> EncodeFeatureMatrixForModelCreation(Integer indexToEncode) { //0-indexed
        if (featureMatrix == null)
            return null;

        //check if this index was already encoded, and if it was just return with error message
        if(encodingDetails.containsKey(indexToEncode)) {
            return null;
        }

        ArrayList<ArrayList<String>> encodedData = new ArrayList<>();
        HashMap<String, ArrayList<Integer>> encoding = new HashMap<>();

        HashSet<String> uniqueKeys = new HashSet<>();
        //Scan the column to collect the unique values
        for (ArrayList<String> dataRow : featureMatrix) {
            String key = dataRow.get(indexToEncode);
            uniqueKeys.add(key);
        }

        Integer index = 0;
        for (String u : uniqueKeys) {
            //Initialize with all 0s
            ArrayList<Integer> encodedValue = new ArrayList<Integer>(Collections.nCopies(uniqueKeys.size(), 0));

            //Set the index alone to hot
            encodedValue.set(index, 1);
            index++;

            //save this encoding, so you can use it
            encoding.put(u, encodedValue);
        }


        for (ArrayList<String> dataRow : featureMatrix) {
            ArrayList<String> encodedRow = new ArrayList<>();
            for (int col = 0; col < dataRow.size(); col++) {
                HashMap<String, ArrayList<Integer>> existingEncoding = encodingDetails.get(col);
                if(existingEncoding != null) {      //This col already has an encoding
                    List<Integer> en = null;
                    en = existingEncoding.get(dataRow.get(col));
                    ArrayList<String> stringList = new ArrayList<>();
                    if(en != null) {
                        for (Integer i : en)
                            stringList.add(i.toString());

                        encodedRow.addAll(stringList);
                    }
                    else
                        encodedRow.add(dataRow.get(col));       //This case would never happen
                } else {
                    if(indexToEncode == col) {     //This is the col that is to be encoded now
                        ArrayList<Integer> currentEncoding = encoding.get(dataRow.get(col));
                        ArrayList<String> stringList = new ArrayList<>();
                        if(currentEncoding != null) {
                            for (Integer i : currentEncoding)
                                stringList.add(i.toString());

                            encodedRow.addAll(stringList);
                        }
                        else
                            encodedRow.add(dataRow.get(col));       //This case would never happen
                    } else  //This is not the col that is to be encoded now, nor does this col have an existing encoding, so just copy the original value
                        encodedRow.add(dataRow.get(col));
                }

            }
            encodedData.add(encodedRow);
        }

        encodedFeatureMatrix = encodedData;

        //append fill encoding details of this call
        //Pair<Integer, HashMap<String, Integer>> detail = new Pair(indexToEncode, encoding);
        encodingDetails.put(indexToEncode, encoding);

        return new Pair(encodedData, encoding);
    }
}
