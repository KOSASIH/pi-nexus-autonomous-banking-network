import React, { useState, useEffect } from 'eact';
import axios from 'axios';

function App() {
    const [nodeList, setNodeList] = useState([]);
    const [newNodeAddress, setNewNodeAddress] = useState('');

    useEffect(() => {
        axios.get('https://api.pi-network.com/get_node_list')
           .then(response => {
                setNodeList(response.data.node_list);
            })
           .catch(error => {
                console.error(error);
            });
    }, []);

    const handleRegisterNode = async () => {
        try {
            const response = await axios.post('https://api.pi-network.com/register_node', { node_address: newNodeAddress });
            setNodeList([...nodeList, newNodeAddress]);
            setNewNodeAddress('');
        } catch (error) {
            console.error(error);
        }
    };

    return (
        <div>
            <h1>Autonomous PI Network</h1>
            <ul>
                {nodeList.map(node => (
                    <li key={node}>{node}</li>
                ))}
            </ul>
            <input type="text" value={newNodeAddress} onChange={e => setNewNodeAddress(e.target.value)} />
            <button onClick={handleRegisterNode}>Register Node</button>
        </div>
    );
}

export default App;
