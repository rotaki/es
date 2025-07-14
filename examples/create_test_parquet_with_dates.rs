//! Create a test Parquet file with Date32 and Decimal128 types

use arrow::array::{Date32Array, Decimal128Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::fs::File;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_records = 1000;
    let filename = "test_dates_decimals.parquet";

    println!("Creating test Parquet file with {} records...", num_records);

    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("product", DataType::Utf8, false),
        Field::new("price", DataType::Decimal128(10, 2), false),
        Field::new("order_date", DataType::Date32, false),
        Field::new("ship_date", DataType::Date32, false),
    ]));

    // Generate data
    let order_ids: Vec<i64> = (0..num_records).map(|i| i as i64).collect();
    let products: Vec<String> = (0..num_records)
        .map(|i| format!("Product_{}", i % 100))
        .collect();

    // Prices as Decimal128 (10.00 to 1009.99)
    let prices: Vec<i128> = (0..num_records)
        .map(|i| (1000 + i * 100) as i128) // Represents 10.00 to 1009.99 when scale=2
        .collect();

    // Dates as days since 1970-01-01
    let base_date = 19000; // Approximately 2022-01-01
    let order_dates: Vec<i32> = (0..num_records)
        .map(|i| base_date + (i as i32 % 365))
        .collect();
    let ship_dates: Vec<i32> = (0..num_records)
        .map(|i| base_date + (i as i32 % 365) + 5) // Ship 5 days after order
        .collect();

    let order_id_array = Int64Array::from(order_ids);
    let product_array = StringArray::from(products);
    let price_array = Decimal128Array::from(prices).with_precision_and_scale(10, 2)?;
    let order_date_array = Date32Array::from(order_dates);
    let ship_date_array = Date32Array::from(ship_dates);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(order_id_array),
            Arc::new(product_array),
            Arc::new(price_array),
            Arc::new(order_date_array),
            Arc::new(ship_date_array),
        ],
    )?;

    let file = File::create(filename)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    println!("Created {} with {} records", filename, num_records);
    println!("Schema includes Int64, Utf8, Decimal128(10,2), and Date32 types");

    Ok(())
}
