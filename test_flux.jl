using Flux

input = rand(28) |> gpu
label = rand(10) |> gpu 

data = [(input, label), (input, label)]

model = Chain(
  Dense(28 => 10)
) |> gpu

model(input)

opt_state = Flux.setup(Adam(), model)

for epoch in 1:100
  Flux.train!(model, data, opt_state) do m, x, y
    Flux.mse(m(x), y)
  end
end