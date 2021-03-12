package com.navis.mlengine.entities;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;
import java.io.Serializable;
import java.util.Date;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Entity
public class MLModel implements Serializable {
    @Id
    @Column(name = "consumer_id")
    private String consumerId;

    @Column(name = "mape")
    private Double mape;

    @Column(name = "rmse")
    private Double rms;

    @Column(name = "accuracy_percentage")
    private Double accuracyPercentage;

    @Column(name = "testRows")
    private Integer testRows;

    @Column(name = "date_of_creation")
    private Date dateOfModelCreation;

    @Column(name = "problem_type")
    private String predictionType;

    @Column(name = "algorithm")
    private String algorithm;

    @Column(name = "features")
    private String features;

    @Column(name = "featureTypes")
    private String featureTypes;

    @Column(name = "output")
    private String output;

    @Column(name = "outputType")
    private String outputType;


    @Column(name = "model_dump")
    @Lob
    private byte[] modelDump;
}

