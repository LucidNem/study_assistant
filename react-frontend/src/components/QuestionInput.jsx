import { useState } from "react";

const QuestionInput = ({ onSubmit }) => {
  const [question, setQuestion] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (question.trim()) {
      onSubmit(question);
      setQuestion(""); // καθαρίζει το input
    }
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
      <label htmlFor="question">Ερώτηση:</label>
      <input
        type="text"
        id="question"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Πληκτρολόγησε την ερώτησή σου..."
        style={{ width: "100%", padding: "0.5rem", marginTop: "0.5rem" }}
      />
      <button type="submit" style={{ marginTop: "1rem" }}>
        Υποβολή
      </button>
    </form>
  );
};

export default QuestionInput;
