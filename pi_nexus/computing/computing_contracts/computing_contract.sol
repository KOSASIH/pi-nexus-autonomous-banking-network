pragma solidity ^0.8.0;

contract ComputingContract {
    struct Job {
        string name;
        string input;
        string output;
        address owner;
        bool completed;
    }

    mapping(address => Job[]) private jobs;

    event JobSubmitted(address indexed owner, uint256 indexed jobId, string name, string input);

    function submitJob(string memory _name, string memory _input) public {
        Job memory newJob;
        newJob.name = _name;
        newJob.input = _input;
        newJob.output = '';
        newJob.owner = msg.sender;
        newJob.completed = false;

        uint256 jobId = jobs[msg.sender].length;
        jobs[msg.sender].push(newJob);

        emit JobSubmitted(msg.sender, jobId, _name, _input);
    }

    function getJob(address _owner, uint256 _jobId) public view returns (string memory, string memory, string memory) {
        Job storage job = jobs[_owner][_jobId];
        return (job.name, job.input, job.output);
    }

    function completeJob(address _owner, uint256 _jobId, string memory _output) public {
        Job storage job = jobs[_owner][_jobId];
        job.output = _output;
        job.completed = true;
    }
}
