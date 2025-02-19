import React, { useState } from 'react';

const FeedbackForm = () => {
    const [feedback, setFeedback] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        // Logic to send feedback to the server
        console.log('Feedback submitted:', feedback);
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2>Report an Issue</h2>
            <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Describe the issue..."
                required
            />
            <button type="submit">Submit</button>
        </form>
    );
};

export default FeedbackForm;
