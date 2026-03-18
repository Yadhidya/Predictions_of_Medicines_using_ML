import axios from "axios";

const API_URL = "http://127.0.0.1:8000/predict";

export const getPrediction = async (data) => {
  try {
    const response = await axios.post(API_URL, {
      date: data.Date,
      Medicine: data.Medicine,
    });
    return response.data; // should be { prediction: 3.32 }
  } catch (error) {
    console.error("Prediction API error:", error);
    return { prediction: null }; // prevent crash
  }
};
