import React from "react";

const Dropdown = ({ label, options, value, onChange }) => {
  return (
    <div className="flex flex-col mb-4">
      <label className="mb-1 font-semibold">{label}</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="border p-2 rounded"
      >
        <option value="">Select {label}</option>
        {options.map(opt => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    </div>
  );
};

export default Dropdown;
