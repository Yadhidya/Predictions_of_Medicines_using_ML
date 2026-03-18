import React, { useState, useEffect } from "react";
import Dropdown from "../components/Dropdown";
import countryCityMap from "../data/country_city_map.json";
import cityProductClassMap from "../data/city_productclass_map.json";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  BarElement,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  BarElement,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend
);

const Home = () => {
  const [country, setCountry] = useState("");
  const [city, setCity] = useState("");
  const [productClass, setProductClass] = useState("");
  const [month, setMonth] = useState("");
  const [year, setYear] = useState("");
  const [cities, setCities] = useState([]);
  const [productClasses, setProductClasses] = useState([]);
  const [errors, setErrors] = useState({});

  const [monthlyPredictions, setMonthlyPredictions] = useState([]);
  const [seasonalityData, setSeasonalityData] = useState([]);

  const [totalSales, setTotalSales] = useState(0);
  const [avgSales, setAvgSales] = useState(0);
  const [disease, setDisease] = useState("");
  const [alert, setAlert] = useState("");
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState("monthly");

  const months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
  ];

  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: currentYear - 2017 }, (_, i) => 2018 + i);

  useEffect(() => {
    setCities(country ? countryCityMap[country] : []);
    setCity("");
    setProductClass("");
    setProductClasses([]);
    setMonthlyPredictions([]);
    setSeasonalityData([]);
    setErrors({});
  }, [country]);

  useEffect(() => {
    setProductClasses(city ? cityProductClassMap[city] : []);
    setProductClass("");
    setMonthlyPredictions([]);
    setSeasonalityData([]);
    setErrors({});
  }, [city]);

  const validateFields = (mode) => {
    const newErrors = {};
    if (!country) newErrors.country = "Country required";
    if (!city) newErrors.city = "City required";
    if (!productClass) newErrors.productClass = "Product class required";
    if (mode === "monthly" && !month) newErrors.month = "Month required";
    if (!year) newErrors.year = "Year required";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handlePredict = async () => {
    if (!validateFields("monthly")) return;
    setLoading(true);
    setMonthlyPredictions([]);
    setSeasonalityData([]);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict_sales", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          country,
          city,
          product_class: productClass,
          month: months.indexOf(month) + 1,
          year: Number(year),
        }),
      });

      const data = await res.json();

      setMonthlyPredictions(data.predictions || []);
      setTotalSales(data.total_predicted_sales || 0);
      setAvgSales(data.average_sales_per_product || 0);
      setDisease(data.disease_category || "N/A");
      setAlert(data.outbreak_alert || "No Alert");
    } catch {
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const handleSeasonality = async () => {
    if (!validateFields("seasonality")) return;
    setLoading(true);
    setSeasonalityData([]);
    setMonthlyPredictions([]);

    try {
      const res = await fetch("http://127.0.0.1:8000/seasonality_analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          country,
          city,
          product_class: productClass,
          month: 1,
          year: Number(year),
        }),
      });

      const data = await res.json();
      setSeasonalityData(data.seasonality || []);
    } catch {
      alert("Seasonality fetch failed");
    } finally {
      setLoading(false);
    }
  };

  const monthlyBarData = {
    labels: monthlyPredictions.map((p) => p.product_name),
    datasets: [
      {
        label: "Predicted Sales (₹)",
        data: monthlyPredictions.map((p) => p.predicted_sales),
        backgroundColor: "rgba(37,99,235,0.8)",
      },
    ],
  };

  const seasonalityLineData = {
    labels: seasonalityData.map(
      (d) => new Date(2025, d.month - 1).toLocaleString("default", { month: "short" })
    ),
    datasets: [
      {
        label: "Total Predicted Sales (₹)",
        data: seasonalityData.map((d) => d.predicted_sales),
        borderColor: "rgba(16,185,129,0.9)",
        backgroundColor: "rgba(16,185,129,0.15)",
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const renderDropdown = (label, options, value, onChange, errorKey) => (
    <div>
      <Dropdown label={label} options={options} value={value} onChange={onChange} />
      {errors[errorKey] && (
        <p className="text-red-600 text-xs mt-1">{errors[errorKey]}</p>
      )}
    </div>
  );

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-3xl font-bold text-center mb-8 text-blue-700">
        Predictive Modelling of Drug Consumption Pattern
      </h1>

      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setViewMode("monthly")}
          className={`px-5 py-2 rounded font-semibold ${
            viewMode === "monthly" ? "bg-blue-600 text-white" : "bg-gray-200"
          }`}
        >
          Monthly Prediction
        </button>
        <button
          onClick={() => setViewMode("seasonality")}
          className={`px-5 py-2 rounded font-semibold ${
            viewMode === "seasonality" ? "bg-green-600 text-white" : "bg-gray-200"
          }`}
        >
          Yearly Seasonality
        </button>
      </div>

      {viewMode === "monthly" && (
        <>
          <div className="grid md:grid-cols-2 gap-4">
            {renderDropdown("Country", Object.keys(countryCityMap), country, setCountry, "country")}
            {renderDropdown("City", cities, city, setCity, "city")}
            {renderDropdown("Product Class", productClasses, productClass, setProductClass, "productClass")}
            {renderDropdown("Month", months, month, setMonth, "month")}
            {renderDropdown("Year", years, year, setYear, "year")}
          </div>

          <button onClick={handlePredict} className="btn-primary mt-6 w-full">
            {loading ? "Predicting..." : "Predict Monthly Sales"}
          </button>

          {monthlyPredictions.length > 0 && (
            <>
              <Bar data={monthlyBarData} className="mt-8" />
              <p className="mt-4 font-semibold">
                Total: ₹{totalSales} | Avg/Product: ₹{avgSales}
              </p>
              <p className="mt-2">
                Disease: <b>{disease}</b> | Alert: <b>{alert}</b>
              </p>
            </>
          )}
        </>
      )}

      {viewMode === "seasonality" && (
        <>
          <div className="grid md:grid-cols-2 gap-4">
            {renderDropdown("Country", Object.keys(countryCityMap), country, setCountry, "country")}
            {renderDropdown("City", cities, city, setCity, "city")}
            {renderDropdown("Product Class", productClasses, productClass, setProductClass, "productClass")}
            {renderDropdown("Year", years, year, setYear, "year")}
          </div>

          <button onClick={handleSeasonality} className="btn-success mt-6 w-full">
            {loading ? "Analyzing..." : "View Seasonality"}
          </button>

          {seasonalityData.length > 0 && (
            <Line data={seasonalityLineData} className="mt-8" />
          )}
        </>
      )}
    </div>
  );
};

export default Home;
