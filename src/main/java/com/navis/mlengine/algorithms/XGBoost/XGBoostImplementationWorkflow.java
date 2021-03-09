package com.navis.mlengine.algorithms.XGBoost;

import com.navis.mlengine.algorithms.BasePredictorWorkflow;
import com.navis.mlengine.configuration.GenericAlgorithmConfigurationBundle;
import com.navis.mlengine.configuration.XGBoostConfigurationBundle;
import com.navis.mlengine.entities.ActualVsPredictions;
import com.navis.mlengine.entities.MLModel;
import com.navis.mlengine.enums.EAlgorithm;
import com.navis.mlengine.enums.EFeatureType;
import com.navis.mlengine.enums.EPredictionType;
import com.navis.mlengine.mlhelpers.encoders.Encoders;
import com.navis.mlengine.mlhelpers.encoders.LabelEncoder;
import com.navis.mlengine.mlhelpers.encoders.OneHotEncoder;
import com.navis.mlengine.service.MLModelEncodingService;
import com.navis.mlengine.service.MLModelService;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.javers.common.collections.Pair;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.io.*;
import java.util.*;

@Slf4j
@Component
@Getter
@Setter
@NoArgsConstructor
public class XGBoostImplementationWorkflow extends BasePredictorWorkflow {
    MLModel mlModel = null;
    private MLModelService mlModelService;
    private MLModelEncodingService mlModelEncodingService;
    private LabelEncoder labelEncoder;
    private OneHotEncoder oneHotEncoder;
    protected ArrayList<ArrayList<String>> encodedFeatureMatrix;
    protected ArrayList<String> encodedClassValues;
    private Encoders encoders;
    private XGBoostConfigurationBundle xgBoostConfigurationBundle;
    private float[] curatedTrainingData;
    private float[] curatedTrainingClasses;
    private float[] curatedValidationData;
    private float[] curatedValidationClasses;
    private float[] curatedTestData;
    private float[] curatedTestClasses;
    private DMatrix trainingMatrix;
    private DMatrix validationMatrix;
    private DMatrix testMatrix;
    Integer rows;
    Integer trainingRowsCount;
    Integer validationRowsCount;
    Integer testRowsCount;
    Integer trainingRowsEndIndex;
    Integer validationRowsEndIndex;
    Integer testRowsEndIndex ;

    //Constructor used for predicting from models\
    /*TO CONTINUE
    public XGBoostImplementationWorkflow(MLModel mlModel, String consumerId) throws Exception {
        this.mlModel = mlModelService.getModelForConsumer(consumerId);
        if(mlModel == null) {
            log.error("Model not found for consumer :" +consumerId);
            throw new Exception("Model not found for consumer :" + consumerId);
            return;
        }
        this.mlModelService = inMlModelService;
        this.mlModelEncodingService = inMlModelEncodingService;
        //Take the raw data and meta data and seperate it out into the appropriate fields in BasePredictorWorkflow, to make it appropriate to work on it

        encoders = new Encoders(super.featureMatrix, super.classValues);
//        encoders.LabelEncodeMatrixForModelCreation(2);
//        encoders.OneHotEncodeFeatureMatrixForModelCreation(4);
//        encoders.LabelEncodeMatrixForModelCreation(1);
//        encoders.OneHotEncodeFeatureMatrixForModelCreation(5);


    }*/

    //Constructor used for creating models
    public XGBoostImplementationWorkflow(GenericAlgorithmConfigurationBundle bundle, MLModelService inMlModelService, MLModelEncodingService inMlModelEncodingService) {
        this.xgBoostConfigurationBundle = (XGBoostConfigurationBundle)bundle;
        this.mlModelService = inMlModelService;
        this.mlModelEncodingService = inMlModelEncodingService;
        //Take the raw data and meta data and seperate it out into the appropriate fields in BasePredictorWorkflow, to make it appropriate to work on it
        this.seperateFeatureAndClass(xgBoostConfigurationBundle);
        encoders = new Encoders(super.featureMatrix, super.classValues);
//        encoders.LabelEncodeMatrixForModelCreation(2);
//        encoders.OneHotEncodeFeatureMatrixForModelCreation(4);
//        encoders.LabelEncodeMatrixForModelCreation(1);
//        encoders.OneHotEncodeFeatureMatrixForModelCreation(5);


    }

    private Boolean encodeDataForInput() {
        if(featureMatrix == null || featureTypes == null) {
            return false;
        }

        try {
            //Curate feature data
            Integer featureIndex = 0;
            for (EFeatureType fT : featureTypes) {
                System.out.println("Looking at index" + featureIndex);

                if (fT == EFeatureType.UNKNOWN)
                    return false;

                if (fT == EFeatureType.STRING || fT == EFeatureType.CATEGORY) {
                    encoders.OneHotEncodeFeatureMatrixForModelCreation(featureIndex);
                } else if (fT == EFeatureType.BOOLEAN) {
                    encoders.BooleanEncodeFeatureMatrixForModelCreation(featureIndex);
                }

                featureIndex++;
            }

            //Filll up the encoded data of feature matrix in class data member
            encodedFeatureMatrix = encoders.getEncodedFeatureMatrix();

            //Curate class data
            if (classType == EFeatureType.STRING || classType == EFeatureType.CATEGORY) {
                encoders.LabelEncodeClassValuesForModelCreation();
            }

            //Filll up the encoded data of class values in class data member
            if(encoders.getEncodedClassValues() != null)
                encodedClassValues = encoders.getEncodedClassValues();
            else
                encodedClassValues = this.classValues;

            //Now we have the features and classes encoded so that it can be fed to XGBoost, fill it up into the data that is to be fed
            rows = encoders.getEncodedFeatureMatrix().size();
            trainingRowsCount = (int)(rows*70*0.01); //Use 70% as training data
            validationRowsCount = (int)(rows*20*0.01); //Use 20% as validation data
            testRowsCount = rows - (trainingRowsCount + validationRowsCount); //Use remaining 10% as test data
            trainingRowsEndIndex = trainingRowsCount;
            validationRowsEndIndex = trainingRowsCount + validationRowsCount;
            testRowsEndIndex = trainingRowsCount + validationRowsCount + testRowsCount;

            //We know that atleast one row exists for training, so get(0) is safe
            Integer cols = encoders.getEncodedFeatureMatrix().get(0).size();
            curatedTrainingData = new float[trainingRowsCount * cols];
            curatedValidationData = new float[validationRowsCount * cols];
            curatedTestData = new float[testRowsCount * cols];

            FileWriter trainingDataFileClassinBeg = new FileWriter("c:\\tmp\\test\\trainingDataWithClassInBeg.txt", false);
            FileWriter trainingDataFileClassinEnd = new FileWriter("c:\\tmp\\test\\trainingDataWithClassInEnd.txt", false);

            Integer validationMatrixIndex = 0, trainingMatrixIndex = 0, testMatrixIndex = 0;
            Boolean printedStart=false, printedEnd=false;
            for (Integer rowIter = 0; rowIter < rows; rowIter++) {
                trainingDataFileClassinBeg.write(this.encodedClassValues.get(rowIter)+",");
                for (Integer colIter = 0; colIter < cols; colIter++) {
                    try {
                        if(rowIter < trainingRowsEndIndex) {
                            curatedTrainingData[trainingMatrixIndex] = Float.valueOf(encodedFeatureMatrix.get(rowIter).get(colIter));
                            trainingDataFileClassinBeg.write(curatedTrainingData[trainingMatrixIndex] +",");
                            trainingDataFileClassinEnd.write(curatedTrainingData[trainingMatrixIndex] +",");
                            trainingMatrixIndex++;
                        } else if (rowIter < validationRowsEndIndex) {
                            curatedValidationData[validationMatrixIndex] = Float.valueOf(encodedFeatureMatrix.get(rowIter).get(colIter));
                            trainingDataFileClassinBeg.write(curatedTrainingData[validationMatrixIndex] +",");
                            trainingDataFileClassinEnd.write(curatedTrainingData[validationMatrixIndex] +",");
                            validationMatrixIndex++;
                        } else if (rowIter < testRowsEndIndex) {
                            if(!printedStart) {
                                ArrayList<ArrayList<String> >input = this.getXgBoostConfigurationBundle().getRawTrainingData();
                                System.out.println("TestMatrix are rows from : "+rowIter+" with details of : "+input.get(rowIter).get(0)+" with class as : "+input.get(rowIter).get(input.get(rowIter).size()-1));
                                printedStart = true;
                            }
                            if(!printedEnd && rowIter==testRowsEndIndex-1) { //Last time
                                ArrayList<ArrayList<String>>input = this.getXgBoostConfigurationBundle().getRawTrainingData();
                                System.out.println("TestMatrix are rows to : "+rowIter+" with details of : "+input.get(rowIter).get(0)+" with class as : "+input.get(rowIter).get(input.get(rowIter).size()-1));
                                printedEnd = true;
                            }
                            curatedTestData[testMatrixIndex] = Float.valueOf(encodedFeatureMatrix.get(rowIter).get(colIter));
                            trainingDataFileClassinBeg.write(curatedTrainingData[testMatrixIndex] +",");
                            trainingDataFileClassinEnd.write(curatedTrainingData[testMatrixIndex] +",");
                            testMatrixIndex++;
                        }
                    } catch(Exception ex) {
                        System.out.println("Exception in filling input matrix for XGBoost");
                        log.error("Exception in filling input matrix for XGBoost", ex);
                    }
                }
                trainingDataFileClassinEnd.write(this.encodedClassValues.get(rowIter)+"\n");
                trainingDataFileClassinBeg.write("\n");
            }
            trainingDataFileClassinBeg.close();


            debugDump();

            //Set the training matrix
            trainingMatrix = new DMatrix(curatedTrainingData, trainingRowsCount, cols);
            validationMatrix = new DMatrix(curatedValidationData, validationRowsCount, cols);
            testMatrix = new DMatrix(curatedTestData, testRowsCount, cols);

            Integer classValueSize = encoders.getClassValues().size();   //Will be the same as the number of training samples, or size of feature matrix
            curatedTrainingClasses = new float[trainingRowsCount];
            curatedValidationClasses = new float[validationRowsCount];
            curatedTestClasses = new float[testRowsCount];

            Integer validationClassIndex = 0, trainingClassIndex = 0, testClassIndex = 0;
            for (Integer classValueIter = 0; classValueIter < rows; classValueIter++) {
                if(classValueIter < trainingRowsEndIndex) {
                    curatedTrainingClasses[trainingClassIndex] = Float.valueOf(encodedClassValues.get(classValueIter));
                    trainingClassIndex++;
                } else if(classValueIter < validationRowsEndIndex) {
                    curatedValidationClasses[validationClassIndex] = Float.valueOf(encodedClassValues.get(classValueIter));
                    validationClassIndex++;
                } else if(classValueIter < testRowsEndIndex) {
                    curatedTestClasses[testClassIndex] = Float.valueOf(encodedClassValues.get(classValueIter));
                    testClassIndex++;
                }
            }

            //Set the labels
            trainingMatrix.setLabel(curatedTrainingClasses);
            validationMatrix.setLabel(curatedValidationClasses);
        } catch(XGBoostError xgBoostException) {
            return false;
        } catch(Exception ex) {
            return false;
        }
        return true;
    }

    private void debugDump() {
        try {
            //testing
            //raw data
            FileWriter rawDataFile = new FileWriter("c:\\tmp\\test\\rawData.txt", false);
            ArrayList<ArrayList<String>> rawD = this.getXgBoostConfigurationBundle().getRawTrainingData();
            for (ArrayList<String> r : rawD) {
                for (String col : r) {
                    rawDataFile.write(col + ",");
                }
                rawDataFile.write("\n");
            }
            rawDataFile.close();

            //training data split
            FileWriter rawSplitTrainingDataFile = new FileWriter("c:\\tmp\\test\\rawSplitTrainingDataFile.txt", false);
            for (ArrayList<String> featureMatrixRow : featureMatrix) {
                for (String col : featureMatrixRow) {
                    rawSplitTrainingDataFile.write(col + ",");
                }
                rawSplitTrainingDataFile.write("\n");
            }
            rawSplitTrainingDataFile.close();
        } catch(IOException ex) {
            System.out.println("Exception in debug dump!");
        }

    }

    private Map<String, Object> getParams() {
        Map<String, Object> params = new HashMap<String, Object>();
        EPredictionType predictionType = this.xgBoostConfigurationBundle.getPredictionType();
        if(predictionType == EPredictionType.MULTICLASS_CLASSIFICATION)
            params.put("objective", "multi:softmax");
        if(predictionType == EPredictionType.BINARY_CLASSIFICATION)
            params.put("objective", "binary:hinge");
        if(predictionType == EPredictionType.REGRESSION) {
            params.put("objective", this.xgBoostConfigurationBundle.getObjective());
            params.put("eval_metric", this.xgBoostConfigurationBundle.getEval_metric());
            params.put("booster", this.xgBoostConfigurationBundle.getBooster());
            params.put("max_depth", this.xgBoostConfigurationBundle.getMax_depth());
            params.put("subsample", this.xgBoostConfigurationBundle.getSubsample());
            params.put("silent", this.xgBoostConfigurationBundle.getSilent());
            params.put("max_delta_step", this.xgBoostConfigurationBundle.getMax_delta_step());
            params.put("min_child_weight", this.xgBoostConfigurationBundle.getMax_child_wt());
            params.put("eta", this.xgBoostConfigurationBundle.getEta());
            params.put("alpha", this.xgBoostConfigurationBundle.getAlpha());
            params.put("lambda", this.xgBoostConfigurationBundle.getLambda());
            params.put("gamma", this.xgBoostConfigurationBundle.getGamma());
            params.put("n_estimators", this.xgBoostConfigurationBundle.getN_estimators());

            params.put("max_depth", 15);
            params.put("max_depth", 15);


            params.put("colsample_bylevel", 0.41010395885331385);
            params.put("colsample_bynode", 0.7277257431773251);
            params.put("colsample_bytree", 0.9328679988478339);
            params.put("gamma", 15.789979674352436);
            params.put("learning_rate", 0.6701479482689346);
            params.put("max_delta_step", 8);
            params.put("max_depth", 15);
            params.put("min_child_weight", 15);
            params.put("n_estimators", 107);
            params.put("num_parallel_tree", 19);
            params.put("refresh_leaf", 1);
            params.put("reg_alpha", 12.683381839585065);
            params.put("reg_lambda", 17.8083918396064);
            params.put("scale_pos_weight", 12.64191256300771);
            params.put("subsample", 0.07789122913725656);
            params.put("tree_method", "auto");

        }
/*

        params.put("verbosity", this.xgBoostConfigurationBundle.getVerbosity());
        params.put("eta", this.xgBoostConfigurationBundle.getEta());
        params.put("alpha", this.xgBoostConfigurationBundle.getAlpha());
        params.put("lambda", this.xgBoostConfigurationBundle.getLambda());
        params.put("gamma", this.xgBoostConfigurationBundle.getGamma());
        params.put("num_class", this.getEncoders().getClassValueEncodingDetails().size());
*/
        System.out.println("The parameters are : verbosity : "+ this.xgBoostConfigurationBundle.getVerbosity()
                +", eta = " + this.xgBoostConfigurationBundle.getEta()
                +", alpha = " + this.xgBoostConfigurationBundle.getAlpha()
                +", lambda = " +this.xgBoostConfigurationBundle.getLambda()
                +", gamma = " +  this.xgBoostConfigurationBundle.getGamma()
                +", num_rounds = " + this.xgBoostConfigurationBundle.getRounds()
                +", num_class"+this.getEncoders().getClassValueEncodingDetails().size()
                +"objective"+ this.xgBoostConfigurationBundle.getObjective()
                +"  eval_metric : "+ this.xgBoostConfigurationBundle.getEval_metric()
                +"booster: "+ this.xgBoostConfigurationBundle.getBooster()
                +"max_depth : "+ this.xgBoostConfigurationBundle.getMax_depth()
                +"subsample : "+ this.xgBoostConfigurationBundle.getSubsample()
                +"silent : "+ this.xgBoostConfigurationBundle.getSilent()
                +"max_delta_step : "+ this.xgBoostConfigurationBundle.getMax_delta_step()
                +"min_child_weight : "+ this.xgBoostConfigurationBundle.getMax_child_wt()
                +"eta : "+ this.xgBoostConfigurationBundle.getEta()
                +"alpha : "+ this.xgBoostConfigurationBundle.getAlpha()
                +"lambda : "+ this.xgBoostConfigurationBundle.getLambda()
                +"gamma : "+ this.xgBoostConfigurationBundle.getGamma()
                +"n_estimators : "+ this.xgBoostConfigurationBundle.getN_estimators());

        return params;
    }

    private Boolean saveEncodingsForModel() {
        return true;
    }

    private MLModel createAndSavePersistableXGBoostModel(Booster booster, List<String> features, Double rms, Double mape, Double percentageAccuracy, Integer testRowsCount) {
        MLModel mlModel = new MLModel();
        try {
            mlModel.setRms(rms);
            mlModel.setMape(mape);
            mlModel.setAccuracyPercentage(percentageAccuracy);
            mlModel.setTestRows(testRowsCount);
            mlModel.setConsumerId(this.getXgBoostConfigurationBundle().getConsumerId());
            mlModel.setPredictionType(this.getXgBoostConfigurationBundle().getPredictionType().name());
            mlModel.setAlgorithm(EAlgorithm.XGBOOST.name());
            mlModel.setDateOfModelCreation(new Date());
            mlModel.setFeatures(features.subList(0,features.size()-1).toString());
            mlModel.setFeatureTypes(super.featureTypes.subList(0,featureTypes.size()).toString());

            mlModel.setOutput(features.get(features.size()-1));
            mlModel.setOutputType(super.featureTypes.get(featureTypes.size()-1).name());

            ByteArrayOutputStream b = new ByteArrayOutputStream();
            booster.saveModel(b);
            mlModel.setModelDump(b.toByteArray());

            this.mlModelService.saveModel(mlModel);

            saveEncodingsForModel();

        } catch (XGBoostError xgBoostError) {
            xgBoostError.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return mlModel;
    }

    @Override
    public Pair<ActualVsPredictions, MLModel> buildAndSaveModel(GenericAlgorithmConfigurationBundle mlBundle) {
    //public MLModel buildAndSaveModel(GenericAlgorithmConfigurationBundle mlBundle) {
        try {
            //Check if the raw data has rows to train data, all columns have a datatype, and all rows are of the same size
            if (!elementaryCheck(mlBundle)) {
                log.error("Elementary check failed!");
                return null;
            }

            //Curates the data and initializes the trainingMatrix with its class labels
            Boolean curationSuccessful = encodeDataForInput();
            if(!curationSuccessful)
                return null;

            Map<String, Object> params = getParams();

            // Specify a watch list to see model accuracy on data sets
            Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
                {
                    put("train", trainingMatrix);
                    put("test", validationMatrix);
                }
            };

            Booster booster = XGBoost.train(trainingMatrix, params, this.xgBoostConfigurationBundle.getRounds(), watches, null, null);

            List<Float> retPredictions = new ArrayList<>();
            List<Float> retActuals = new ArrayList();

            //Calculate the percentage accuracy on 10% of the data given
            float predictions[][] = booster.predict(testMatrix);
            ArrayList<Boolean> predictionEvaluation = new ArrayList<>();
            Integer correctPredictions = 0, total = 0;
            Double summation = 0.0;
            Double mape_summation = 0.0;
            for(float f[] : predictions) {
                retPredictions.add(f[0]);
                retActuals.add(curatedTestClasses[total]);
                if(f[0] == curatedTestClasses[total]) {
                    Integer predictionIndex = total+trainingRowsCount + validationRowsCount;
                    ArrayList<String> predictedElementDetails = this.getXgBoostConfigurationBundle().getRawTrainingData().get(predictionIndex);
                    String predictedValueDecoded = encoders.getClassValueEncodingDetails().inverse().get((int)f[0]);
                    System.out.println("Correctly Predicted the outcome of " + predictedElementDetails.toString() + " as encoded value '" + f[0] + "' decoded value : '" + predictedValueDecoded + "'");
                    correctPredictions++;
                } else {
                    Double diff = Double.valueOf(curatedTestClasses[total] - f[0]);

                    //MAPE
                    Double thisTerm = diff/curatedTestClasses[total];
                    mape_summation += Math.abs(thisTerm);

                    System.out.println("Actual = "+ curatedTestClasses[total]+", Prediction = "+f[0]+", Difference = "+diff+", Error = "+((f[0]-curatedTestClasses[total])/curatedTestClasses[total])*100+"%");
                    Double diffSquare = Math.pow(diff,2);
                    summation += diffSquare;
                }
                total++;
            }
            Double percentageAccuracy=0.0, rms=0.0, mape=0.0;
            if(this.getXgBoostConfigurationBundle().getPredictionType() == EPredictionType.MULTICLASS_CLASSIFICATION ||
                    this.getXgBoostConfigurationBundle().getPredictionType() == EPredictionType.BINARY_CLASSIFICATION) {
                percentageAccuracy = ((Double.valueOf( correctPredictions) / Double.valueOf(total))) * 100.0;
                System.out.println("The accuracy percentage as calculated on " + testRowsCount + "(10% of the given data) rows is " + percentageAccuracy);

            } else {
                rms = Math.sqrt(summation / total);
                System.out.println("RMS as calculated on " + testRowsCount + "(10% of the given data) rows is " + rms);

                mape = mape_summation / total;
                System.out.println("MAPE  as calculated on " + testRowsCount + "(10% of the given data) rows is " + mape + "%");
            }

            createAndSavePersistableXGBoostModel(booster, mlBundle.getFeatureNames(), rms, mape, percentageAccuracy, testRowsCount);

            ActualVsPredictions results = new ActualVsPredictions();
            results.setActuals(retActuals);
            results.setPredictions(retPredictions);

            Pair<ActualVsPredictions, MLModel> ret = new Pair(results,mlModel);

            System.out.println("Created model!");

            return ret;
        } catch (XGBoostError xgBoostError) {
            System.out.println("xgboost exception");

        } catch (Exception ex) {
            System.out.println("exception");
        }

        return null;
    }

    public MLModel createModel() {
        //mlAlgorithmConfigurationBundle
        return null;
    }



    DMatrix testMat;
    DMatrix valMat;
    //LobHelper lobHelper;

    //EntityManager entityManager;

    //public XGBoostImplementationWorkflow(EntityManager  e) {
    //  entityManager = e;
    //}

    //  public XGBoostImplementationWorkflow() {

    //}




    public MLModel createXGBoostModel() {
        return null;
    }

    @Transactional
    void setModel(MLModel model, ByteArrayOutputStream b) {
        //Session session = entityManager.unwrap(Session.class);
        // SessionFactory sessionFactory = (new SessionFactoryBean()).getSessionFactory();
        // Session session = sessionFactory.getCurrentSession();


        //Blob blob = session.getLobHelper().createBlob(b.toByteArray());


        //model.setModelDump(b.toByteArray());
    }

    public void loadModelAndPredict(MLModel model) {
        Booster booster = null;

        try {
            //InputStream i = new InputStream(model.getModelDump());
            InputStream modelStream = new ByteArrayInputStream(model.getModelDump());

            booster = XGBoost.loadModel(modelStream);
        } catch (IOException e) {
            e.printStackTrace();
        } catch(Exception ex) {
            ex.getStackTrace();
        }



        try {
            // predict
            testMat = new DMatrix("c:\\tmp\\satimage.scale.t");
            float[][] predicts = booster.predict(testMat);


            // predict leaf
            float[][] leafPredicts = booster.predictLeaf(testMat, 0);
            System.out.println("Prediction done!");
        } catch (XGBoostError xgBoostError) {
            xgBoostError.printStackTrace();
        }


    }

/*working function

    //@Override
    public MLModel buildModel_orig(GenericAlgorithmConfigurationBundle mlBundle) {
        //this.setXgBoostConfigurationBundle((XGBoostConfigurationBundle) mlBundle);
       // Map<String, Object> params = getParams();
/*
        try {
            //trainMat = new DMatrix("c:\\tmp\\satimage.scale.tr");
            float[] d = new float[100*10] ;
            float[] l = new float[100] ;

            for(int row =0;row <100*10;row++) {


                d[row]=row*2;

            }

            trainMat = new DMatrix(d,100,10);
            trainMat.setLabel(l);



            int i=100, j=10;
            valMat = new DMatrix(d,i ,j);

            valMat = new DMatrix("c:\\tmp\\satimage.scale.val");

            if(valMat == null || trainMat == null) {
                log.error("Could not construct the validation matrix or training matrix, so aborting!");
                return null;
            }
        } catch (XGBoostError xgBoostError) {
            xgBoostError.printStackTrace();
        }

        // Specify a watch list to see model accuracy on data sets
        Map<String, DMatrix> watches = new HashMap<String, DMatrix>() {
            {
                put("train", trainMat);
                put("test", valMat);
            }
        };
        int nround = 2;
        try {
            Booster booster = XGBoost.train(trainMat, params, nround, watches, null, null);

 */
/*
            ByteArrayOutputStream b = new ByteArrayOutputStream();
            booster.saveModel("c:\\tmp\\model.bin");

            booster.saveModel(b);

 */
    // SerialBlob blob = new SerialBlob(b.toByteArray());




    /*

                    MLModel model = new MLModel();
                    model.setAlgorithm("XGBOOST");
                    model.setDateOfModelCreation(new Date());
                    model.setFeatures("a,c,b");
                    //works - model.setModelDump(b.toByteArray());


                    System.out.println("really-----------!");
                    return model;





                } catch (XGBoostError xgBoostError) {
                    xgBoostError.printStackTrace();
                    return null;
                } catch (Exception ex) {
                    log.error("Exception!");
                    return null;
                }


            }
        */
    @Override
    public MLModel predictFromModel() {
        return null;
    }
}