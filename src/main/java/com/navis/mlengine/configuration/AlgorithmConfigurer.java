package com.navis.mlengine.configuration;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.navis.mlengine.enums.EAlgorithm;
import com.navis.mlengine.enums.EFeatureType;
import com.navis.mlengine.enums.EPredictionType;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.io.Resource;
import org.springframework.util.ResourceUtils;


import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Map;

@Configuration
@Getter
@Setter
@Slf4j
public class AlgorithmConfigurer {

    private GenericAlgorithmConfigurationBundle mlAlgorithmConfiguration;

    @Autowired
    private ApplicationContext context;

    private static final String CONSUMERS_NODE = "consumers";
    private static final String ML_CONFIG_FILE = "classpath:mlengine.json";

    public GenericAlgorithmConfigurationBundle InitializeMLEngineForPredictionFromModel(String consumerId, ArrayList<ArrayList<String>> inPredictionDataRecd)
    {
        byte[] jsonData = new byte[0];
        JsonNode mlConfiguration = null;

        /*try {
            File mlConfigurationFile = ResourceUtils.getFile(ML_CONFIG_FILE);*/
            Resource resource = new ClassPathResource("mlengine.json");
            InputStream inputStream = null;
            try {
                inputStream = resource.getInputStream();
                Reader reader = new InputStreamReader(inputStream, "UTF-8");
                jsonData = IOUtils.toByteArray(reader);
                //        jsonData = Files.readAllBytes(Paths.get(mlConfigurationFile.getAbsolutePath()));
                mlConfiguration =  (new ObjectMapper()).readTree(jsonData);
            } catch (UnsupportedEncodingException exception) {
                exception.printStackTrace();
                return null;

            } catch (IOException exception) {
                exception.printStackTrace();

                exception.printStackTrace();
                return null;


            }
        //Get the configuration of the consumerId
        JsonNode configurationOfThisConsumerId = (mlConfiguration.path(CONSUMERS_NODE)).path(consumerId);

        if(configurationOfThisConsumerId.isMissingNode()) //Configuration for this consumer does not exist
            return null;

        try {
            JsonNode activeAlgorithmNode = configurationOfThisConsumerId.path("algorithm");
            EAlgorithm activeAlgorithm = (new ObjectMapper()).convertValue(activeAlgorithmNode, EAlgorithm.class);

            JsonNode predictionTypeNode = configurationOfThisConsumerId.path("predictionType");
            EPredictionType predictionType = (new ObjectMapper()).convertValue(predictionTypeNode, EPredictionType.class);

            JsonNode featuresNode = configurationOfThisConsumerId.path("features");
            if(featuresNode.isMissingNode())
                return null;

            Map<String, EFeatureType> featuresMap = (new ObjectMapper()).convertValue(featuresNode, new TypeReference<Map<String, EFeatureType>>() {});
            ArrayList<EFeatureType> featureTypes = new ArrayList<EFeatureType>(featuresMap.values());
            ArrayList<String> featureNames = new ArrayList<String>(featuresMap.keySet());

            JsonNode activeAlgorithmParamNode = configurationOfThisConsumerId.path("algoparams").get(activeAlgorithm.name());

            mlAlgorithmConfiguration = createBundleForPrediction(consumerId, activeAlgorithm, predictionType, featureTypes, featureNames, inPredictionDataRecd, activeAlgorithmParamNode);
        } catch (IllegalArgumentException ex) {
            log.error("Exception1!");
            return null;
        } catch(Exception ex) {
            log.error("Exception2!");
            return null;
        }

        return mlAlgorithmConfiguration;
    }

    public GenericAlgorithmConfigurationBundle InitializeMLEngineForModelCreation(String consumerId, ArrayList<ArrayList<String>> trainingData) {
        byte[] jsonData = new byte[0];
        JsonNode mlConfiguration;

        try {
            File mlConfigurationFile = ResourceUtils.getFile(ML_CONFIG_FILE);
            jsonData = Files.readAllBytes(Paths.get(mlConfigurationFile.getAbsolutePath()));
            mlConfiguration =  (new ObjectMapper()).readTree(jsonData);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        //Get the configuration of the consumerId
        JsonNode configurationOfThisConsumerId = (mlConfiguration.path(CONSUMERS_NODE)).path(consumerId);

        if(configurationOfThisConsumerId.isMissingNode()) //Configuration for this consumer does not exist
            return null;

        try {
            JsonNode activeAlgorithmNode = configurationOfThisConsumerId.path("algorithm");
            EAlgorithm activeAlgorithm = (new ObjectMapper()).convertValue(activeAlgorithmNode, EAlgorithm.class);

            JsonNode predictionTypeNode = configurationOfThisConsumerId.path("predictionType");
            EPredictionType predictionType = (new ObjectMapper()).convertValue(predictionTypeNode, EPredictionType.class);

            JsonNode featuresNode = configurationOfThisConsumerId.path("features");
            if(featuresNode.isMissingNode())
                return null;

            Map<String, EFeatureType> featuresMap = (new ObjectMapper()).convertValue(featuresNode, new TypeReference<Map<String, EFeatureType>>() {});
            ArrayList<EFeatureType> featureTypes = new ArrayList<EFeatureType>(featuresMap.values());
            ArrayList<String> featureNames = new ArrayList<String>(featuresMap.keySet());

            JsonNode activeAlgorithmParamNode = configurationOfThisConsumerId.path("algoparams").get(activeAlgorithm.name());

            mlAlgorithmConfiguration = createBundle(consumerId, activeAlgorithm, predictionType, featureTypes, featureNames, trainingData, activeAlgorithmParamNode);
        } catch (IllegalArgumentException ex) {
            log.error("Exception1!");
            return null;
        } catch(Exception ex) {
            log.error("Exception2!");
            return null;
        }

        return mlAlgorithmConfiguration;
    }

    private GenericAlgorithmConfigurationBundle createBundleForPrediction(String consumerId, EAlgorithm inAlgorithm, EPredictionType inPredictionType,
                                                             ArrayList<EFeatureType> inFeatureTypes,
                                                             ArrayList<String> inFeatureNames,
                                                             ArrayList<ArrayList<String>> inDataForPrediction,
                                                             JsonNode activeAlgoParamNode) {
        if (inAlgorithm == EAlgorithm.XGBOOST) {
            //XGBoostConfigurationBundle xgBoostConfigurationBundle = new XGBoostConfigurationBundle();
            XGBoostConfigurationBundle xgBoostConfigurationBundle = context.getBean(XGBoostConfigurationBundle.class);
            XGBoostConfigurationBundle xgBoostConfigurationBundle1 = context.getBean(XGBoostConfigurationBundle.class);
            xgBoostConfigurationBundle.setGamma(1.2);

            if (!activeAlgoParamNode.isMissingNode())
                xgBoostConfigurationBundle = (new ObjectMapper()).convertValue(activeAlgoParamNode, XGBoostConfigurationBundle.class);

            xgBoostConfigurationBundle.setConsumerId(consumerId);
            xgBoostConfigurationBundle.setFeatureTypesWithoutClassForPrediction(inFeatureTypes);
            //Remove the label in the end - we expect the input to come in without the label(we will add a dummy label)
            int toRemove = xgBoostConfigurationBundle.getFeatureTypesWithoutClassForPrediction().size() - 1;
            xgBoostConfigurationBundle.getFeatureTypesWithoutClassForPrediction().remove(toRemove);
            xgBoostConfigurationBundle.setPredictionType(inPredictionType);
            xgBoostConfigurationBundle.setRawDataForPrediction(inDataForPrediction);

            return xgBoostConfigurationBundle;
        }

        if (inAlgorithm == EAlgorithm.NN) {
            NeuralNetworkConfigurationBundle neuralNetworkConfigurationBundle = new NeuralNetworkConfigurationBundle();

            if (!activeAlgoParamNode.isMissingNode())
                neuralNetworkConfigurationBundle = (new ObjectMapper()).convertValue(activeAlgoParamNode, NeuralNetworkConfigurationBundle.class);

            neuralNetworkConfigurationBundle.setConsumerId(consumerId);
            neuralNetworkConfigurationBundle.setFeatureTypesIncludingClass(inFeatureTypes);
            neuralNetworkConfigurationBundle.setPredictionType(inPredictionType);
            neuralNetworkConfigurationBundle.setRawDataForPrediction(inDataForPrediction);

            return neuralNetworkConfigurationBundle;
        }

        return null;
    }

    private GenericAlgorithmConfigurationBundle createBundle(String consumerId, EAlgorithm inAlgorithm, EPredictionType inPredictionType,
                                                             ArrayList<EFeatureType> inFeatureTypes,
                                                             ArrayList<String> inFeatureNames,
                                                             ArrayList<ArrayList<String>> inTrainingData,
                                                             JsonNode activeAlgoParamNode) {

        if (inAlgorithm == EAlgorithm.XGBOOST) {
            XGBoostConfigurationBundle xgBoostConfigurationBundle = new XGBoostConfigurationBundle();

            if (!activeAlgoParamNode.isMissingNode())
                xgBoostConfigurationBundle = (new ObjectMapper()).convertValue(activeAlgoParamNode, XGBoostConfigurationBundle.class);

            xgBoostConfigurationBundle.setConsumerId(consumerId);
            xgBoostConfigurationBundle.setFeatureTypesIncludingClass(inFeatureTypes);
            xgBoostConfigurationBundle.setFeatureNames(inFeatureNames);
            xgBoostConfigurationBundle.setPredictionType(inPredictionType);
            xgBoostConfigurationBundle.setRawTrainingData(inTrainingData);

            return xgBoostConfigurationBundle;
        }

        if (inAlgorithm == EAlgorithm.NN) {
            NeuralNetworkConfigurationBundle neuralNetworkConfigurationBundle = new NeuralNetworkConfigurationBundle();

            if (!activeAlgoParamNode.isMissingNode())
                neuralNetworkConfigurationBundle = (new ObjectMapper()).convertValue(activeAlgoParamNode, NeuralNetworkConfigurationBundle.class);

            neuralNetworkConfigurationBundle.setConsumerId(consumerId);
            neuralNetworkConfigurationBundle.setFeatureTypesIncludingClass(inFeatureTypes);
            neuralNetworkConfigurationBundle.setPredictionType(inPredictionType);
            neuralNetworkConfigurationBundle.setRawTrainingData(inTrainingData);

            return neuralNetworkConfigurationBundle;
        }

        return null;
    }

}
