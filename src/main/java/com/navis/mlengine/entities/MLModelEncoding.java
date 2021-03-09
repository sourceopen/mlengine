package com.navis.mlengine.entities;

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

    @Column(name = "column_no")
    Integer column;

    @Column(name = "field_name")
    String field;

    @Column(name = "encoding")
    String encoding;
}
