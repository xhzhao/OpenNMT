require 'mklnn'
--[[ Encoder is a unidirectional Sequencer used for the source language.

    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n


Inherits from [onmt.Sequencer](onmt+modules+Sequencer).
--]]
local Encoder, parent = torch.class('onmt.Encoder', 'onmt.Sequencer')

local options = {
  {
    '-layers', 2,
    [[Number of recurrent layers of the encoder and decoder. See also `-enc_layers`, `-dec_layers`
      and `-bridge` to assign different layers to the encoder and decoder.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-rnn_size', 500,
    [[Hidden size of the recurrent unit.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isUInt(),
      structural = 0
    }
  },
  {
    '-rnn_type', 'LSTM',
    [[Type of recurrent cell.]],
    {
      enum = {'LSTM', 'GRU'},
      structural = 0
    }
  },
  {
    '-dropout', 0.3,
    [[Dropout probability applied between recurrent layers.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0, 1),
      structural = 1
    }
  },
  {
    '-dropout_input', false,
    [[Dropout probability applied to the input of the recurrent module.]],
    {
      structural = 0
    }
  },
  {
    '-dropout_words', 0,
    [[Dropout probability applied to the source sequence.]],
    {
      valid = onmt.utils.ExtendedCmdLine.isFloat(0, 1),
      structural = 1
    }
  },
  {
    '-dropout_type', 'naive',
    [[Dropout type.]],
    {
      structural = 0,
      enum = { 'naive', 'variational'}
    }
  },
  {
    '-residual', false,
    [[Add residual connections between recurrent layers.]],
    {
      structural = 0
    }
  }
}

function Encoder.declareOpts(cmd)
  cmd:setCmdLineOptions(options)
end

--[[ Construct an Encoder layer.

Parameters:

  * `inputNetwork` - input module.
  * `rnn` - recurrent module.
]]
function Encoder:__init(args, inputNetwork)
  local RNN = onmt.LSTM
  if args.rnn_type == 'GRU' then
    RNN = onmt.GRU
  end

  local rnn = RNN.new(args.layers, inputNetwork.inputSize, args.rnn_size, args.dropout, args.residual, args.dropout_input, args.dropout_type)

  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numStates = self.rnn.numStates
  self.args.dropout_type = args.dropout_type

  parent.__init(self, self:_buildModel())

  self.mklnnLSTM = mklnn.LSTMFullStep(500, 500):float()

  self:resetPreallocation()
end

--[[ Return a new Encoder using the serialized data `pretrained`. ]]
function Encoder.load(pretrained)
  local self = torch.factory('onmt.Encoder')()

  self.args = pretrained.args
  self.args.numStates = self.args.numStates or self.args.numEffectiveLayers -- Backward compatibility.

  parent.__init(self, pretrained.modules[1])

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function Encoder:serialize()
  return {
    name = 'Encoder',
    modules = self.modules,
    args = self.args
  }
end

function Encoder:resetPreallocation()
  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context vector.
  self.contextProto = torch.Tensor()
end

--[[ Build one time-step of an Encoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t})}$$

  Where $$c^l$$ and $$h^l$$ are the hidden and cell states at each layer,
  $$x_t$$ is a sparse word to lookup.
--]]
function Encoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numStates do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  -- Input word.
  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)

  -- Compute input network.
  local input = self.inputNet(x)
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)
  return nn.gModule(inputs, { outputs })
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states
  2. - context matrix H
--]]
function Encoder:forward(batch, initial_states)

  -- TODO: Change `batch` to `input`.

  local outputSize = self.args.rnnSize

  local states = initial_states
  -- if states is not passed, start with empty state
  if not states then
    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numStates,
                                                           self.stateProto,
                                                           { batch.size, outputSize })
    end

    -- Make initial states h_0.
    states = onmt.utils.Tensor.reuseTensorTable(self.statesProto, { batch.size, outputSize })
  end

    -- Preallocated output matrix.
  local context = onmt.utils.Tensor.reuseTensor(self.contextProto,
                                                { batch.size, batch.sourceLength, outputSize })

  if self.train then
    self.inputs = {}
    if self.args.dropout_type == 'variational' then
      -- Initialize noise for variational dropout.
      onmt.VariationalDropout.initializeNetwork(self.network)
    end
  end

  --xhzhao
  local weight = self.rnn:parameters()
  local wx   = weight[1]
  local wx_b = weight[2]
  local wh   = weight[3]
  local wh_b = weight[4]
  --wx:fill(3)
  --wh:fill(2)
  wx_b:zero()
  wh_b:zero()
  local WETensor = torch.FloatTensor(batch.sourceLength, batch.size,  outputSize)
  local inputs_mklnn = {}

  onmt.utils.Table.append(inputs_mklnn, states)

  local ori_start = sys.clock()
  -- Act like nn.Sequential and call each clone in a feed-forward
  -- fashion.
  for t = 1, batch.sourceLength do

    -- Construct "inputs". Prev states come first then source.
    local inputs = {}
    onmt.utils.Table.append(inputs, states)
    table.insert(inputs, batch:getSourceInput(t))

    --xhzhao
    local we = self.inputNet:forward(inputs[3])
    WETensor[t] = we


    if self.train then
      -- Remember inputs for the backward pass.
      self.inputs[t] = inputs
    end

    states = self:net(t):forward(inputs)

    -- Make sure it always returns table.
    if type(states) ~= "table" then states = { states } end

    -- Zero states of timesteps with padding.
    if batch:variableLengths() then
      for b = 1, batch.size do
        if t <= batch.sourceLength - batch.sourceSize[b] then
          for j = 1, #states do
            states[j][b]:zero()
          end
        end
      end
    end

    -- Copy output (h^L_t = states[#states]) to context.
    context[{{}, t}]:copy(states[#states])
  end
  local ori_end = sys.clock()
--[[
  -- xhzhao code
  print("-----context-----")
  print(context:size())
  print("context sum = ", context:sum())
  print("-----states-----")
  print(states)
  print("states[1]  sum = ", states[1]:sum())
  print("states[2]  sum = ", states[2]:sum())

  print("batch.sourceLength = ",batch.sourceLength)

    print("-----WETensor-----")
    print("WETensor sum = ", WETensor:sum(), "type = ",WETensor:type())
    WETensor_size = WETensor:size()
    local T = WETensor_size[1]
    local N = WETensor_size[2]
    local H = WETensor_size[3]
    print("T = ",T)
    print("N = ",N)
    print("H = ",H)

    --table.insert(inputs_mklnn, WETensor)
    print("-----inputs_mklnn-----")
    print("inputs_mklnn[1] sum = ", inputs_mklnn[1]:sum())
    print("inputs_mklnn[2] sum = ", inputs_mklnn[2]:sum())
    print("inputs_mklnn[3] sum = ", inputs_mklnn[3]:sum())

    print("wx size = ",wx:size())
    print("wx sum  = ",wx:sum())

    local tempWx = wx:clone()
    local tempWh = wh:clone()
    tempWx:resize(4,H,H):transpose(2,3)
    tempWh:resize(4,H,H):transpose(2,3)

    print("wxi sum = ",tempWx[1]:sum())
    print("wxf sum = ",tempWx[2]:sum())
    print("wxo sum = ",tempWx[3]:sum())
    print("wxt sum = ",tempWx[4]:sum())
]]--

    table.insert(inputs_mklnn, WETensor)
    WETensor_size = WETensor:size()
    local T = WETensor_size[1]
    local N = WETensor_size[2]
    local H = WETensor_size[3]
    local temp1 = wx:transpose(1,2)
    self.mklnnLSTM.weightX[1] = temp1[{{}, {1,H}}]
    self.mklnnLSTM.weightX[2] = temp1[{{}, {H+1,2*H}}]
    self.mklnnLSTM.weightX[3] = temp1[{{}, {2*H+1,3*H}}]
    self.mklnnLSTM.weightX[4] = temp1[{{}, {3*H+1,4*H}}]

    local temp2 = wh:transpose(1,2)
    self.mklnnLSTM.weightH[1] = temp2[{{}, {1,H}}]
    self.mklnnLSTM.weightH[2] = temp2[{{}, {H+1,2*H}}]
    self.mklnnLSTM.weightH[3] = temp2[{{}, {2*H+1,3*H}}]
    self.mklnnLSTM.weightH[4] = temp2[{{}, {3*H+1,4*H}}]

--[[
    self.mklnnLSTM.weightX:copy(tempWx)
    self.mklnnLSTM.weightH:copy(tempWh)
    print("self.mklnnLSTM.weightX size = ", self.mklnnLSTM.weightX:size())
    print("self.mklnnLSTM.weightX sum  = ", self.mklnnLSTM.weightX:sum())
    print("self.mklnnLSTM.weightH sum  = ", self.mklnnLSTM.weightH:sum())
]]--
    local mkl_start = sys.clock()
    local output_mklnn = self.mklnnLSTM:forward(inputs_mklnn)
    local mkl_end = sys.clock()
--[[
    print("-----mklnn output-----")
    print(output_mklnn)
    print("output_mklnn c:sum() = ",output_mklnn[2]:sum())
    print("output_mklnn h:sum() = ",output_mklnn[1]:sum())
    print("output_mklnn next_c:sum() = ",output_mklnn[4]:sum())
    print("output_mklnn next_h:sum() = ",output_mklnn[3]:sum())
]]--

  check_1 = torch.all(torch.lt(torch.abs(torch.add(output_mklnn[1]:transpose(1,2), -context)), 1e-6))
  check_2 = torch.all(torch.lt(torch.abs(torch.add(output_mklnn[4], -states[1])), 1e-6))
  check_3 = torch.all(torch.lt(torch.abs(torch.add(output_mklnn[3], -states[2])), 1e-6))
  print("context check = ",check_1, check_2, check_3)
  print("T="..T..", N="..N..", H="..H.." ori_time = ",ori_end-ori_start, " mkl_time = ", mkl_end - mkl_start," ori/mkl = ",(ori_end-ori_start)/(mkl_end - mkl_start))

  return states, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state - this can be null if states are not used
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function Encoder:backward(batch, gradStatesOutput, gradContextOutput)
  -- TODO: change this to (input, gradOutput) as in nngraph.
  local outputSize = self.args.rnnSize
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numStates,
                                                              self.gradOutputProto,
                                                              { batch.size, outputSize })
  end

  local gradStatesInput
  if gradStatesOutput then
    gradStatesInput = onmt.utils.Tensor.copyTensorTable(self.gradOutputsProto, gradStatesOutput)
  else
    -- if gradStatesOutput is not defined - start with empty tensor
    gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto, { batch.size, outputSize })
  end

  local gradInputs = {}

  for t = batch.sourceLength, 1, -1 do
    -- Add context gradients to last hidden states gradients.
    gradStatesInput[#gradStatesInput]:add(gradContextOutput[{{}, t}])

    -- Zero gradients of timesteps with padding.
    if batch:variableLengths() then
      for b = 1, batch.size do
        if t <= batch.sourceLength - batch.sourceSize[b] then
          for j = 1, #gradStatesInput do
            gradStatesInput[j][b]:zero()
          end
        end
      end
    end

    -- nngraph does not accept table of size 1.
    local timestepGradOutput = #gradStatesInput > 1 and gradStatesInput or gradStatesInput[1]

    local gradInput = self:net(t):backward(self.inputs[t], timestepGradOutput)

    -- Prepare next Encoder output gradients.
    for i = 1, #gradStatesInput do
      gradStatesInput[i]:copy(gradInput[i])
    end

    -- Gather gradients of all user inputs.
    gradInputs[t] = {}
    for i = #gradStatesInput + 1, #gradInput do
      table.insert(gradInputs[t], gradInput[i])
    end

    if #gradInputs[t] == 1 then
      gradInputs[t] = gradInputs[t][1]
    end
  end
  -- TODO: make these names clearer.
  -- Useful if input came from another network.
  return gradInputs

end
