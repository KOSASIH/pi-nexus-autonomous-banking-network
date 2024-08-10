import React, { useState, useEffect, useMemo } from 'react';
import { useTable, useSortBy, useFilters } from 'react-table';
import { NodeCard } from './NodeCard';
import { NodeFilters } from './NodeFilters';
import { NodeSort } from './NodeSort';
import { NodePagination } from './NodePagination';
import { NodeActions } from './NodeActions';
import { useNodeData } from '../hooks/useNodeData';
import { useNodeSelection } from '../hooks/useNodeSelection';
import { useNodeIncentivization } from '../hooks/useNodeIncentivization';
import { useNodeReputation } from '../hooks/useNodeReputation';

const NodeList = () => {
  const [nodes, setNodes] = useState([]);
  const [selectedNodes, setSelectedNodes] = useState([]);
  const [incentivizationData, setIncentivizationData] = useState({});
  const [reputationData, setReputationData] = useState({});

  const { data, error, isLoading } = useNodeData();
  const { nodeSelection, setNodeSelection } = useNodeSelection();
  const { nodeIncentivization, setNodeIncentivization } = useNodeIncentivization();
  const { nodeReputation, setNodeReputation } = useNodeReputation();

  useEffect(() => {
    if (data) {
      setNodes(data.nodes);
      setIncentivizationData(data.incentivization);
      setReputationData(data.reputation);
    }
  }, [data]);

  const columns = useMemo(
    () => [
      {
        Header: 'Node Name',
        accessor: 'name',
      },
      {
        Header: 'Reputation',
        accessor: 'reputation',
        Cell: ({ value }) => (
          <div>
            <span>{value}</span>
            <NodeReputationChart data={reputationData} nodeId={value} />
          </div>
        ),
      },
      {
        Header: 'Incentivization',
        accessor: 'incentivization',
        Cell: ({ value }) => (
          <div>
            <span>{value}</span>
            <NodeIncentivizationChart data={incentivizationData} nodeId={value} />
          </div>
        ),
      },
      {
        Header: 'Actions',
        accessor: 'actions',
        Cell: ({ value }) => (
          <NodeActions
            nodeId={value}
            onSelect={(nodeId) => setNodeSelection((prev) => [...prev, nodeId])}
            onDeselect={(nodeId) => setNodeSelection((prev) => prev.filter((id) => id !== nodeId))}
          />
        ),
      },
    ],
    [reputationData, incentivizationData]
  );

  const {
    getTableProps,
    getTheadProps,
    getTrProps,
    getThProps,
    getTdProps,
    headerGroups,
    rows,
    prepareRow,
  } = useTable({
    columns,
    data: nodes,
  });

  const { sortBy, sortDirection } = useSortBy({
    initialState: {
      sortBy: [{ id: 'reputation', desc: true }],
    },
  });

  const { filters, setFilters } = useFilters({
    initialState: {
      filters: [],
    },
  });

  const filteredNodes = useMemo(() => {
    let filteredData = nodes;
    filters.forEach((filter) => {
      filteredData = filteredData.filter((node) => node[filter.id] === filter.value);
    });
    return filteredData;
  }, [nodes, filters]);

  const paginatedNodes = useMemo(() => {
    const startIndex = (nodePagination.currentPage - 1) * nodePagination.pageSize;
    const endIndex = startIndex + nodePagination.pageSize;
    return filteredNodes.slice(startIndex, endIndex);
  }, [filteredNodes, nodePagination]);

  return (
    <div className="node-list">
      <NodeFilters filters={filters} setFilters={setFilters} />
      <NodeSort sortBy={sortBy} sortDirection={sortDirection} />
      <table {...getTableProps()} className="node-table">
        <thead>
          {headerGroups.map((headerGroup) => (
            <tr {...headerGroup.getHeaderGroupProps()} className="node-table-header">
              {headerGroup.headers.map((column) => (
                <th {...column.getHeaderProps()} className="node-table-header-cell">
                  {column.render('Header')}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {paginatedNodes.map((node) => {
            prepareRow(node);
            return (
              <tr {...node.getRowProps()} className="node-table-row">
                {node.cells.map((cell) => (
                  <td {...cell.getCellProps()} className="node-table-cell">                    
                     {cell.render('Cell')}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <NodePagination
        nodes={paginatedNodes}
        currentPage={nodePagination.currentPage}
        pageSize={nodePagination.pageSize}
        totalCount={filteredNodes.length}
        onPageChange={(page) => nodePagination.setCurrentPage(page)}
      />
      <NodeCard
        nodes={selectedNodes}
        onNodeSelect={(nodeId) => setSelectedNodes((prev) => [...prev, nodeId])}
        onNodeDeselect={(nodeId) => setSelectedNodes((prev) => prev.filter((id) => id !== nodeId))}
      />
    </div>
  );
};

export default NodeList;
