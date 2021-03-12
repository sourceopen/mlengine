package com.navis.mlengine.controller;

import com.navis.mlengine.algorithms.BasePredictorWorkflow;
import com.navis.mlengine.algorithms.XGBoost.XGBoostImplementationWorkflow;
import com.navis.mlengine.configuration.AlgorithmConfigurer;
import com.navis.mlengine.configuration.GenericAlgorithmConfigurationBundle;
import com.navis.mlengine.configuration.NeuralNetworkConfigurationBundle;
import com.navis.mlengine.configuration.XGBoostConfigurationBundle;
import com.navis.mlengine.entities.MLModel;
import com.navis.mlengine.entities.ResponseForPredictions;
import com.navis.mlengine.entities.ResponseForTraining;
import com.navis.mlengine.enums.EPredictionType;
import com.navis.mlengine.service.MLModelEncodingService;
import com.navis.mlengine.service.MLModelService;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;
import org.javers.common.collections.Pair;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

@RestController
@AllArgsConstructor
@Slf4j
public class MLEngineClientCommunicationController {

    private final MLModelService mlModelService;
    private final MLModelEncodingService mlModelEncodingService;
    private final AlgorithmConfigurer mlAlgorithmConfigurer;
    BasePredictorWorkflow predictorWorkflow;

    @GetMapping("/api/areyouready")
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody()
    public ResponseEntity<String> getReadyStatus(HttpServletRequest request, HttpServletResponse response) {
        return new ResponseEntity("I am ready!", HttpStatus.OK);
    }

    @PostMapping("/api/PredictUsingModel")
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody()
    public ResponseEntity<ResponseForPredictions> predictUsingModel(@RequestBody ArrayList<ArrayList<String>> inPredictionDataRecd,  @RequestParam(name = "consumerId") String consumerId,
                                                                HttpServletRequest request, HttpServletResponse response) {
        GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle = mlAlgorithmConfigurer.InitializeMLEngineForPredictionFromModel(consumerId, inPredictionDataRecd);

        MLModel model = mlModelService.getModelForConsumer(consumerId);

        if(model == null) {
            log.error("Model does not exist for " + consumerId);
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        List<Float> predictions = new ArrayList<>();

        try {
            predictorWorkflow = this.getAppropriateWorkflowForPrediction(mlAlgorithmConfigurationBundle);
            if(mlAlgorithmConfigurationBundle.getPredictionType() == EPredictionType.REGRESSION)
                predictions = predictorWorkflow.predictFromModelForRegression(model, mlAlgorithmConfigurationBundle);
            else if(mlAlgorithmConfigurationBundle.getPredictionType() == EPredictionType.BINARY_CLASSIFICATION)
                predictions = predictorWorkflow.predictFromModelForBinaryClassification(model, mlAlgorithmConfigurationBundle);
        } catch(Exception ex) {
            log.error("XGBoost error encountered!", ex);
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }

        ResponseForPredictions res = new ResponseForPredictions();

        res.setRmse(model.getRms());
        res.setMape(model.getMape());
        res.setPredictions(predictions);

        return ResponseEntity
                .created(URI.create(String.format("/api/CreateModelWithTrainingData")))
                .body(res);
    }

    @PostMapping("/api/CreateModelWithTrainingData")
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody()
    public ResponseEntity<ResponseForTraining> modelCreationWithTrainingData(@RequestBody ArrayList<ArrayList<String>> inTrainingDataRecd, @RequestParam(name = "consumerId") String consumerId,
                                                                             HttpServletRequest request, HttpServletResponse response) {
        GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle = mlAlgorithmConfigurer.InitializeMLEngineForModelCreation(consumerId, inTrainingDataRecd);

        if(mlAlgorithmConfigurationBundle == null) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        predictorWorkflow = this.getAppropriateWorkflow(mlAlgorithmConfigurationBundle);

        Pair<ResponseForTraining, MLModel> res =  predictorWorkflow.buildAndSaveModel(mlAlgorithmConfigurationBundle);

        return ResponseEntity
                    .created(URI.create(String.format("/api/CreateModelWithTrainingData")))
                    .body(res.left());
    }

    private BasePredictorWorkflow getAppropriateWorkflow(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        if(mlAlgorithmConfigurationBundle instanceof XGBoostConfigurationBundle)
            return new XGBoostImplementationWorkflow(mlAlgorithmConfigurationBundle, mlModelService, mlModelEncodingService);
        if(mlAlgorithmConfigurationBundle instanceof NeuralNetworkConfigurationBundle)
            return null;    //Not yet implemented

        return null;
    }

    private BasePredictorWorkflow getAppropriateWorkflowForPrediction(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        if(mlAlgorithmConfigurationBundle instanceof XGBoostConfigurationBundle)
            return new XGBoostImplementationWorkflow(mlAlgorithmConfigurationBundle, mlModelEncodingService);
        if(mlAlgorithmConfigurationBundle instanceof NeuralNetworkConfigurationBundle)
            return null;    //Not yet implemented

        return null;
    }
}
