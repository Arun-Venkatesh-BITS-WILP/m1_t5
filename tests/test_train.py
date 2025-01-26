import os
from src.train import train_model
from src.predict import predict


def test_train_model():
    # Train model
    train_model()

    # Check if model file is created
    assert os.path.exists("models/cnn_model.h5"), "Model file not created!"


def test_inference_model():
    # Predict model
    img_path = os.path.join("test_data", "airplane.jpg")
    model_result = predict(img_path)
    expected_result = "airplane"

    # Assert if the predicted result matches the expected result
    assert model_result == expected_result, (
        f"Test failed: Expected '{expected_result}', but got '{model_result}'"
    )

    print(f"Test passed: The predicted class is '{model_result}'")


if __name__ == "__main__":
    test_train_model()
