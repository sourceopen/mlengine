package com.navis.mlengine.controller;

import com.navis.mlengine.algorithms.BasePredictorWorkflow;
import com.navis.mlengine.algorithms.XGBoost.XGBoostImplementationWorkflow;
import com.navis.mlengine.configuration.AlgorithmConfigurer;
import com.navis.mlengine.configuration.GenericAlgorithmConfigurationBundle;
import com.navis.mlengine.configuration.NeuralNetworkConfigurationBundle;
import com.navis.mlengine.configuration.XGBoostConfigurationBundle;
import com.navis.mlengine.entities.ActualVsPredictions;
import com.navis.mlengine.entities.MLModel;
import com.navis.mlengine.service.MLModelEncodingService;
import com.navis.mlengine.service.MLModelService;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.javers.common.collections.Pair;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.net.URI;
import java.util.ArrayList;

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
/* TO CONTINUE
    @PostMapping("/api/PredictUsingModel")
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody()
    public ResponseEntity<String> predictUsingModel(@RequestBody ArrayList<ArrayList<String>> inPredictionDataRecd,  @RequestParam(name = "consumerId") String consumerId,
                                                                HttpServletRequest request, HttpServletResponse response) {
        GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle = mlAlgorithmConfigurer.InitializeMLEngineForPredictionFromModel(consumerId, inPredictionDataRecd);

        MLModel model = mlModelService.getModelForConsumer(consumerId);

        if(model == null) {
            log.error("Model does not exist for " + consumerId);
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        predictorWorkflow = this.getAppropriateWorkflow(model, consumerId);

        predictorWorkflow.buildAndSaveModel(mlAlgorithmConfigurationBundle);
        return new ResponseEntity<>(HttpStatus.OK);


    }
*/
        @PostMapping("/api/CreateModelWithTrainingData")
    @ResponseStatus(HttpStatus.OK)
    @ResponseBody()
    public ResponseEntity<ActualVsPredictions> modelCreationWithTrainingData(@RequestBody ArrayList<ArrayList<String>> inTrainingDataRecd, @RequestParam(name = "consumerId") String consumerId,
                                                                            HttpServletRequest request, HttpServletResponse response) {
        GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle = mlAlgorithmConfigurer.InitializeMLEngineForModelCreation(consumerId, inTrainingDataRecd);

        if(mlAlgorithmConfigurationBundle == null) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }

        predictorWorkflow = this.getAppropriateWorkflow(mlAlgorithmConfigurationBundle);

        Pair<ActualVsPredictions, MLModel> t =  predictorWorkflow.buildAndSaveModel(mlAlgorithmConfigurationBundle);

      //  return new ResponseEntity<>(HttpStatus.OK);

            return ResponseEntity
                    .created(URI
                            .create(String.format("/api/CreateModelWithTrainingData")))
                    .body(t.left());


/*

        long startTime = System.nanoTime();

        String[][] newArray = new String[trainingDataRecd.size()][];
        try {
            int iter = 0;
            for (iter = 0; iter < trainingDataRecd.size(); iter++) {
                newArray[iter] = trainingDataRecd.get(iter).stream().toArray(String[]::new);
            }
        }catch(Exception ex) {
            System.out.println("Exception!");
            System.out.println("Exception!");
        }

        //String display = a.Display(newArray);
        //XGBoostImplementationWorkflow xgBoostImplementation = new XGBoostImplementationWorkflow(entityManager);
        XGBoostImplementationWorkflow xgBoostImplementation = new XGBoostImplementationWorkflow(mlModelService, mlModelEncodingService);
        System.out.println("The number of models available now are : "+(mlModelService.getAllModels()).size());
        MLModel m = xgBoostImplementation.createXGBoostModel();
        mlModelService.saveModel(m);
        List<MLModel> modelList = mlModelService.getAllModels();
        System.out.println("The number of models available now are : "+modelList.size());
        xgBoostImplementation.loadModelAndPredict(modelList.get(0));
*/
    }

    private BasePredictorWorkflow getAppropriateWorkflow(GenericAlgorithmConfigurationBundle mlAlgorithmConfigurationBundle) {
        if(mlAlgorithmConfigurationBundle instanceof XGBoostConfigurationBundle)
            return new XGBoostImplementationWorkflow(mlAlgorithmConfigurationBundle, mlModelService, mlModelEncodingService);
        if(mlAlgorithmConfigurationBundle instanceof NeuralNetworkConfigurationBundle)
            return null;    //Not yet implemented

        return null;
    }
/*
    private BasePredictorWorkflow getAppropriateWorkflow(MLModel mlModel, String consumerId) {
        if(EAlgorithm.valueOf(mlModel.getAlgorithm()).equals(EAlgorithm.XGBOOST))
            return new XGBoostImplementationWorkflow(mlModel, consumerId);
        else if(EAlgorithm.valueOf(mlModel.getAlgorithm()).equals(EAlgorithm.NN))
            return null;    //Not yet implemented

        return null;
    }*/

}
