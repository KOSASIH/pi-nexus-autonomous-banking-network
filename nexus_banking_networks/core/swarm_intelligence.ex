defmodule SwarmIntelligence do
  use GenServer

  def start_link() do
    GenServer.start_link(__MODULE__, [])
  end

  def init([]) do
    {:ok, []}
  end

  def handle_cast({:add_agent, agent}, state) do
    {:noreply, [agent | state]}
  end

  def handle_cast({:remove_agent, agent}, state) do
    {:noreply, List.delete(state, agent)}
  end

  def handle_call(:get_agents, _from, state) do
    {:reply, state, state}
  end
end

defmodule Agent do
  use GenServer

  def start_link() do
    GenServer.start_link(__MODULE__, [])
  end

  def init([]) do
    {:ok, []}
  end

  def handle_cast({:update_resource, resource}, state) do
    {:noreply, [resource | state]}
  end

  def handle_call(:get_resource, _from, state) do
    {:reply, state, state}
  end
end

# Example usage
SwarmIntelligence.start_link()
Agent.start_link()

SwarmIntelligence.add_agent(Agent)
SwarmIntelligence.add_agent(Agent)

Agent.update_resource("Resource 1")
Agent.update_resource("Resource 2")

SwarmIntelligence.get_agents()
|> Enum.each(fn agent ->
  Agent.get_resource(agent)
  |> IO.inspect()
end)
