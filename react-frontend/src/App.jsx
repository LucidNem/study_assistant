import { useState } from "react";
import CourseSelector from "./components/CourseSelector";
import QuestionInput from "./components/QuestionInput";

function App() {
  const [selectedCourse, setSelectedCourse] = useState("");
  const [userQuestion, setUserQuestion] = useState("");

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Virtual Study Assistant</h1>

      <CourseSelector
        selectedCourse={selectedCourse}
        onCourseChange={setSelectedCourse}
      />

      {selectedCourse && (
        <>
          <p style={{ marginTop: "1rem" }}>
            Επιλέχθηκε μάθημα: <strong>{selectedCourse}</strong>
          </p>

          <QuestionInput onSubmit={setUserQuestion} />

          {userQuestion && (
            <p style={{ marginTop: "1rem", fontStyle: "italic" }}>
              ➤ Ερώτηση που δόθηκε: {userQuestion}
            </p>
          )}
        </>
      )}
    </div>
  );
}

export default App;
