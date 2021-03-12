package com.navis.mlengine.mlhelpers.encoders;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.navis.mlengine.entities.MLModelEncoding;
import com.navis.mlengine.enums.EEncodingType;
import com.navis.mlengine.service.MLModelEncodingService;
import javafx.util.Pair;
import lombok.Getter;
import lombok.Setter;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.*;

@Getter
@Setter
public class Encoders {
    //featureMatrixEncodingDetails is encodings (in the featureMatrix - can be label encoded or one hot encoded) for use constructing the test data -
    // Intgeger in the pair corresponds to the column(0 indexed) which was encoded
    //and hashmap contains value->encoding. encoding can be integer(for label encoding) or list<integer> for one hot encoding
    // List of the whole thing becuase we can have many columns encoded
    @Autowired
    private MLModelEncodingService mlModelEncodingService;
    private HashMap<Integer, HashMap<String, Object>> featureMatrixEncodingDetails = new HashMap<>();

    //encodedClassValues is encoding (in the classValues - can only be label encoded) for reconstruction -
    //the hashmap contains value->encoding. encoding can be integer only(only label encoding can be done for class values)
    private BiMap<String, Integer> classValueEncodingDetails = HashBiMap.create();

    private ArrayList<ArrayList<String>> featureMatrix;
    private ArrayList<String> classValues;
    private ArrayList<ArrayList<String>> encodedFeatureMatrix;
    private ArrayList<String> encodedClassValues;

    private ArrayList<ArrayList<String>> matrixForPrediction;
    private ArrayList<ArrayList<String>> encodedDataToPredictMatrix;

    public Encoders(ArrayList<ArrayList<String>> inTestMatrix) {
        matrixForPrediction = inTestMatrix;

        /*ArrayList<String> encodedClassValuesCopy = inClassValues;
        for (String e : encodedClassValuesCopy) {
            encodedClassValues.add(e.replace("\n", "").replace("\r", ""));

        }*/
    }

    public Encoders(ArrayList<ArrayList<String>> inFeatureMatrix, ArrayList<String> inClassValues) {
        featureMatrix = inFeatureMatrix;
        classValues = inClassValues;
        encodedFeatureMatrix = inFeatureMatrix;
        encodedClassValues = new ArrayList<>();

        ArrayList<String> encodedClassValuesCopy = inClassValues;
        for (String e : encodedClassValuesCopy) {
            encodedClassValues.add(e.replace("\n", "").replace("\r", ""));

        }
    }

    public Pair<ArrayList<String>, HashMap<String, Integer>> LabelEncodeClassValuesForModelCreation() {
        if(classValues == null)
            return null;

        if(classValueEncodingDetails.size() > 0)    //can be encoded only once, so return if it is already encoded
            return null;

        ArrayList<String> encodedData = new ArrayList<>();
        BiMap<String, Integer> encoding = HashBiMap.create();
        Integer maxEncode = 0;

        for(String data : classValues) {
            Integer value = encoding.get(data);
            if(value != null) {
                encodedData.add(value.toString());       //Put is as a string later will be converted before passing to algorithm as the last stage
            } else {
                encoding.put(data, maxEncode);
                encodedData.add(maxEncode.toString());
                maxEncode++;
            }
        }

        encodedClassValues = encodedData;
        classValueEncodingDetails = encoding;

        return new Pair(encodedData, encoding);
    }

    public Pair<ArrayList<ArrayList<String>>, HashMap<String, Integer>> LabelEncodeMatrixForModelCreation(Integer indexToEncode) { //0-indexed
        if(featureMatrix == null)
            return null;

        //check if this index was already encoded, and if it was just return with error message
        if(featureMatrixEncodingDetails.containsKey(indexToEncode)) {
            return null;
        }

        ArrayList<ArrayList<String>> encodedData = new ArrayList<>();
        HashMap<String, Integer> encoding = new HashMap<>();
        Integer maxEncode = 1;

        for(ArrayList<String> dataRow : featureMatrix) {
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
                    HashMap<String, Object> existingEncoding = featureMatrixEncodingDetails.get(col);
                    if(existingEncoding != null) {      //This col already has an encoding
                        if(existingEncoding.values().size() > 0 && existingEncoding.values().toArray()[0] instanceof List) {
                                ArrayList<Integer> en = ((ArrayList) existingEncoding.get(dataRow.get(col)));
                                ArrayList<String> stringList = new ArrayList<>();
                                if (en != null) {
                                    for (Integer i : en)
                                        stringList.add(i.toString());

                                    encodedRow.addAll(stringList);
                                } else
                                    encodedRow.add(dataRow.get(col));       //This case would never happen
                        } else { //it is label encoding
                            Integer en = ((Integer) existingEncoding.get(dataRow.get(col)));
                            encodedRow.add(en.toString());
                        }
                    } else {    //Not the col to encode, nor does it already have encoding, just copy from the original
                        encodedRow.add(dataRow.get(col));
                    }
                }
            }
            encodedData.add(encodedRow);
        }

        encodedFeatureMatrix = encodedData;

        //append fill encoding details of this call
        featureMatrixEncodingDetails.put(indexToEncode, new HashMap<String, Object> (encoding));

        return new Pair(encodedData, encoding);
    }

    /*public Pair<ArrayList<ArrayList<String>>, HashMap<String, Integer>> EncodeForPrediction() {
    }*/

    public Pair<ArrayList<ArrayList<String>>, HashMap<String, ArrayList<Integer>>> BooleanEncodeFeatureMatrixForModelCreation(Integer indexToEncode) { //0-indexed
        if (featureMatrix == null)
            return null;

        //check if this index was already encoded, and if it was just return with error message
        if(featureMatrixEncodingDetails.containsKey(indexToEncode)) {
            return null;
        }

        ArrayList<ArrayList<String>> encodedData = new ArrayList<>();
        HashMap<String, Integer> encoding = new HashMap<>();

        //save this encoding, so you can use it
        encoding.put("true", 1);
        encoding.put("false", 0);

        encoding.put("t", 1);
        encoding.put("f", 0);

        for (ArrayList<String> dataRow : featureMatrix) {
            ArrayList<String> encodedRow = new ArrayList<>();
            for (int col = 0; col < dataRow.size(); col++) {
                HashMap<String, Object> existingEncoding = featureMatrixEncodingDetails.get(col);
                if(existingEncoding != null) {      //This col already has an encoding
                    if(existingEncoding.values().size() > 0 && existingEncoding.values().toArray()[0] instanceof List) {  //it is one hot encoding
                        ArrayList<Integer> en = ((ArrayList) existingEncoding.get(dataRow.get(col)));
                        ArrayList<String> stringList = new ArrayList<>();
                        if (en != null) {
                            for (Integer i : en)
                                stringList.add(i.toString());

                            encodedRow.addAll(stringList);
                        } else {
                            encodedRow.add(dataRow.get(col));       //This case would never happen
                        }
                    } else { //it is label encoding
                        Integer en = ((Integer) existingEncoding.get(dataRow.get(col)));
                        encodedRow.add(en.toString());
                    }
                } else {
                    if(indexToEncode == col) {     //This is the col that is to be encoded now
                        String valueToEncode = dataRow.get(col);
                        String encodedValue = "0";
                        encodedRow.add(encoding.get(valueToEncode).toString());
                    } else {    //This is not the col that is to be encoded now, nor does this col have an existing encoding, so just copy the original value
                        encodedRow.add(dataRow.get(col));
                    }
                }
            }
            encodedData.add(encodedRow);
        }

        encodedFeatureMatrix = encodedData;

        //append fill encoding details of this call
        featureMatrixEncodingDetails.put(indexToEncode, new HashMap<String, Object> (encoding));

        return new Pair(encodedData, encoding);
    }

    public Pair<ArrayList<ArrayList<String>>, HashMap<String, ArrayList<Integer>>> OneHotEncodeFeatureMatrixForModelCreation(Integer indexToEncode, String consumerId) { //0-indexed
        if (featureMatrix == null)
            return null;

        //check if this index was already encoded, and if it was just return with error message
        if(featureMatrixEncodingDetails.containsKey(indexToEncode)) {
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

        List<MLModelEncoding> mlModelEncodingList = new ArrayList<>();
        Integer index = 0;
        for (String u : uniqueKeys) {
            //Initialize with all 0s
            ArrayList<Integer> encodedValue = new ArrayList<Integer>(Collections.nCopies(uniqueKeys.size(), 0));

            //Set the index alone to hot
            encodedValue.set(index, 1);

            //save this encoding, so you can use it
            encoding.put(u, encodedValue);

            MLModelEncoding mlModelEncoding = new MLModelEncoding();
            mlModelEncoding.setConsumerId(consumerId);
            mlModelEncoding.setEncodingType(EEncodingType.OHE);
            mlModelEncoding.setColumnNumber(indexToEncode);
            mlModelEncoding.setField(u);
            mlModelEncoding.setTotalUniqueValues(uniqueKeys.size());
            mlModelEncoding.setHotNumber(index);
            mlModelEncodingList.add(mlModelEncoding);

            //For the next one
            index++;
        }

        //save to db
        mlModelEncodingService.addAll(mlModelEncodingList);

        for (ArrayList<String> dataRow : featureMatrix) {
            ArrayList<String> encodedRow = new ArrayList<>();
            for (int col = 0; col < dataRow.size(); col++) {
                HashMap<String, Object> existingEncoding = featureMatrixEncodingDetails.get(col);
                if(existingEncoding != null) {      //This col already has an encoding
                    if(existingEncoding.values().size() > 0 && existingEncoding.values().toArray()[0] instanceof List) {
                            ArrayList<Integer> en = ((ArrayList) existingEncoding.get(dataRow.get(col)));
                            ArrayList<String> stringList = new ArrayList<>();
                            if (en != null) {
                                for (Integer i : en)
                                    stringList.add(i.toString());

                                encodedRow.addAll(stringList);
                            } else {
                                //1

                                //1
                                encodedRow.add(dataRow.get(col));       //This case would never happen
                            }
                    } else { //it is label encoding
                        Integer en = ((Integer) existingEncoding.get(dataRow.get(col)));
                        encodedRow.add(en.toString());
                    }
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
                    } else {    //This is not the col that is to be encoded now, nor does this col have an existing encoding, so just copy the original value
                        encodedRow.add(dataRow.get(col));
                    }
                }
            }
            encodedData.add(encodedRow);
        }

        encodedFeatureMatrix = encodedData;

        //append fill encoding details of this call
        featureMatrixEncodingDetails.put(indexToEncode, new HashMap<String, Object> (encoding));

        return new Pair(encodedData, encoding);
    }

}
