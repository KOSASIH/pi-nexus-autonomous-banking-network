import React, { useState } from 'react';

const Form = ({ onSubmit }) => {
    const [input, setInput] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(input);
        setInput('');
    };

    return (
        <form onSubmit={handleSubmit}>
            <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter your input"
            />
            <button type="submit">Submit</button>
        </form>
    );
};

export default Form;
