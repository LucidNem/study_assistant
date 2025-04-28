import { useState } from "react";

const CourseSelector = ({ selectedCourse, onCourseChange }) => {
  const courses = ["OS", "DSA", "AI", "Math", "Networks"];

  return (
    <div>
      <label htmlFor="course-select">Μάθημα:</label>
      <select
        id="course-select"
        value={selectedCourse}
        onChange={(e) => onCourseChange(e.target.value)}
      >
        <option value="">-- Επιλέξτε μάθημα --</option>
        {courses.map((course) => (
          <option key={course} value={course}>
            {course}
          </option>
        ))}
      </select>
    </div>
  );
};

export default CourseSelector;
