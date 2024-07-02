const formatData = (data) => {
  return data.map((item) => ({
    label: item.label,
    value: item.value,
  }));
};

export default formatData;
