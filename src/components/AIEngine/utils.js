export const preprocessData = (data) => {
    // Implement data preprocessing logic here
    return data.map(item => ({
        input: item.input,
        label: item.label
    }));
};
