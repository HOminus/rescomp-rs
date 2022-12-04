use nalgebra::DMatrix;
use rescomp::{
    activation_function::ActivationFunctionWrapper,
    echo_state_network::EchoStateNetworkBuilder,
    input_projection::{DefaultInputProjection, InputProjectionWithEmbedding},
    reservoir::training::ReservoirTraining,
    state_measurement::DefaultStateMeasurement,
    Reservoir,
};

#[test]
#[cfg_attr(miri, ignore)]
fn esn_predict_sine_cosine() {
    let mut esn_builder = EchoStateNetworkBuilder::<f64>::random(200, 6);
    esn_builder.spectral_radius(0.9);
    let esn = esn_builder
        .build_sparse_discrete_network(ActivationFunctionWrapper::new(|_, v: f64| v.tanh()));

    let input_projection = DefaultInputProjection::new_random(2, 200, 1.0);
    let reservoir = Reservoir::new(input_projection, esn);
    let reservoir_state_measurement = DefaultStateMeasurement::<f64>::new(200);

    let mut data = Vec::with_capacity(4000);
    for t in 0..(data.capacity() / 2) {
        let time = t as f64 * 0.02;
        data.push(time.sin());
        data.push(time.cos());
    }
    let train_data = DMatrix::from_vec(2, data.len() / 2, data);

    let mut rt = ReservoirTraining::new(500, 1000, 0, 500);
    rt.add_data(train_data);
    let mut reservoir_computer =
        rt.train_via_ridge_regression(reservoir, reservoir_state_measurement);

    let kickstarter = rt.get_prediction_kickstarter(0, 1);
    let true_prediction = rt.get_true_future(0);

    let prediction = reservoir_computer.synchronize_and_predict(kickstarter, 0, 500);

    println!("{:}", prediction.columns(0, 5));
    println!("{:}", true_prediction.columns(0, 5));

    let mut total_error = 0.0;
    for (prediction, actual) in prediction.column_iter().zip(true_prediction.column_iter()) {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 500_f64
    );

    let mut total_error = 0.0;
    for (prediction, actual) in prediction
        .columns(300, 150)
        .column_iter()
        .zip(true_prediction.columns(300, 150).column_iter())
    {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 150_f64
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn esn_predict_sine_cosine_embedded_no_stride() {
    let mut esn_builder = EchoStateNetworkBuilder::<f64>::random(200, 6);
    esn_builder.spectral_radius(0.9);
    let esn = esn_builder
        .build_sparse_discrete_network(ActivationFunctionWrapper::new(|_, v: f64| v.tanh()));

    let embedded_input_projection = InputProjectionWithEmbedding::<f64>::new_random(2, 200, 4, 1);
    let reservoir = Reservoir::new(embedded_input_projection, esn);
    let reservoir_state_measurement = DefaultStateMeasurement::<f64>::new(200);

    let mut data = Vec::with_capacity(4000);
    for t in 0..(data.capacity() / 2) {
        let time = t as f64 * 0.02;
        data.push(time.sin());
        data.push(time.cos());
    }
    let train_data = DMatrix::from_vec(2, data.len() / 2, data);

    let mut rt = ReservoirTraining::new(500, 1000, 0, 500);
    rt.add_data(train_data);
    let mut reservoir_computer =
        rt.train_via_ridge_regression(reservoir, reservoir_state_measurement);

    let kickstarter = rt.get_prediction_kickstarter(0, 5);
    let true_prediction = rt.get_true_future(0);

    let prediction = reservoir_computer.synchronize_and_predict(kickstarter, 0, 500);

    println!("{:}", prediction.columns(0, 5));
    println!("{:}", true_prediction.columns(0, 5));

    let mut total_error = 0.0;
    for (prediction, actual) in prediction.column_iter().zip(true_prediction.column_iter()) {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 500_f64
    );

    let mut total_error = 0.0;
    for (prediction, actual) in prediction
        .columns(300, 150)
        .column_iter()
        .zip(true_prediction.columns(300, 150).column_iter())
    {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 150_f64
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn esn_predict_sine_cosine_embedded_with_stride() {
    let mut esn_builder = EchoStateNetworkBuilder::<f64>::random(200, 6);
    esn_builder.spectral_radius(0.9);
    let esn = esn_builder
        .build_sparse_discrete_network(ActivationFunctionWrapper::new(|_, v: f64| v.tanh()));

    let embedded_input_projection = InputProjectionWithEmbedding::<f64>::new_random(2, 200, 3, 2);
    let reservoir = Reservoir::new(embedded_input_projection, esn);
    let reservoir_state_measurement = DefaultStateMeasurement::<f64>::new(200);

    let mut data = Vec::with_capacity(4000);
    for t in 0..(data.capacity() / 2) {
        let time = t as f64 * 0.02;
        data.push(time.sin());
        data.push(time.cos());
    }
    let train_data = DMatrix::from_vec(2, data.len() / 2, data);

    let mut rt = ReservoirTraining::new(500, 1000, 0, 500);
    rt.add_data(train_data);
    let mut reservoir_computer =
        rt.train_via_ridge_regression(reservoir, reservoir_state_measurement);

    let kickstarter = rt.get_prediction_kickstarter(0, 7);
    let true_prediction = rt.get_true_future(0);

    let prediction = reservoir_computer.synchronize_and_predict(kickstarter, 0, 500);

    println!("{:}", prediction.columns(0, 5));
    println!("{:}", true_prediction.columns(0, 5));

    let mut total_error = 0.0;
    for (prediction, actual) in prediction.column_iter().zip(true_prediction.column_iter()) {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 500_f64
    );

    let mut total_error = 0.0;
    for (prediction, actual) in prediction
        .columns(300, 150)
        .column_iter()
        .zip(true_prediction.columns(300, 150).column_iter())
    {
        let (s, c) = (actual[0], actual[1]);
        let (ps, pc) = (prediction[0], prediction[1]);
        total_error += (ps - s).abs() + (pc - c).abs();
    }
    println!(
        "Total Error: {total_error} Avg err: {}",
        total_error / 150_f64
    );
}
