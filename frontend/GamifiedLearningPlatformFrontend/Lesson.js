import React, { useState, useEffect } from 'react';

const Lesson = ({ lesson }) => {
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    // Check if the user has already completed the lesson
    blockchainBasicsContract.methods.getProgress(web3.eth.accounts[0]).call().then((progress) => {
      if (progress >= lesson.id) {
        setCompleted(true);
      }
    });
  }, [lesson]);

  const handleCompleteLesson = async () => {
    await completeLesson(lesson.id);
    setCompleted(true);
  };

  return (
    <div>
      <h2>{lesson.title}</h2>
      <p>{lesson.description}</p>
      {completed ? (
        <p>Lesson completed!</p>
      ) : (
        <button onClick={handleCompleteLesson}>Complete Lesson</button>
      )}
    </div>
  );
};

export default Lesson;
