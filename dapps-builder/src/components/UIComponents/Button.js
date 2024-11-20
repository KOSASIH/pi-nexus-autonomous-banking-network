import React from 'react';

const Button = ({ onClick, label, style }) => {
    return (
        <button onClick={onClick} style={style}>
            {label}
        </button>
    );
};

export default Button;
