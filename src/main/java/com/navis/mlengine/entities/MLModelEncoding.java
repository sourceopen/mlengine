package com.navis.mlengine.entities;

import com.navis.mlengine.enums.EEncodingType;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import javax.persistence.*;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Entity
public class MLModelEncoding {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "encoding_id")
    Long id;

    @Column(name = "prediction_label")
    Boolean predictionLabel;

    @Column(name = "encoding_type")
    EEncodingType encodingType;

    @Column(name = "consumer_id")
    String consumerId;

    @Column(name = "column_no")
    Integer columnNumber;

    @Column(name = "field_name")
    String field;

    @Column(name = "total_unique_values")
    Integer totalUniqueValues;

    @Column(name = "hot_number")
    Integer hotNumber;
}
