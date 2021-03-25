### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ bac88c10-6b19-11eb-3368-e180db4a6825
using Distributions, Plots, StatsPlots, KernelFunctions, LinearAlgebra, PlutoUI

# ╔═╡ 3791104a-6b1b-11eb-11d5-8dbda458e0f3
md"""
## 1-Dimensional Gaussian

μ: $(@bind μ Slider(-.5:0.1:5., default=0., show_value=true))    


σ: $(@bind σ Slider(0.1:0.1:5., default=1., show_value=true))
"""

# ╔═╡ ac90924a-6b1a-11eb-2f1c-27fc62ccc345
plot(Normal(μ, σ), xlims=(-5., 5.), ylims=(0., 1.), linewidth=3)

# ╔═╡ 1016fae2-6b1c-11eb-1aa6-71ec793159bd
md"""
2-Dimensional Gaussian
 
ρ: $(@bind ρ Slider(-0.95:0.01:0.95, default=0., show_value=true))
"""

# ╔═╡ f2b9c820-6b1c-11eb-081a-2b1f6fa548f3
begin 
	m = [0., 0.]
	Σ = [1. ρ; ρ 1.]
	mvnorm = MultivariateNormal(m, Σ)
	
	xs = -2.5:0.05:2.5
	ys = -2.5:0.05:2.5
	z = Array{Float64}(undef, size(xs, 1), size(ys, 1))
	for (idx, x) in enumerate(xs)
		for (jdx, y) in enumerate(ys)
			z[idx, jdx] = pdf(mvnorm, [x, y])
		end
	end

	plot(contour(xs, ys, z, levels=10, linewidth=3))
end

# ╔═╡ 4c04d3bc-6bdc-11eb-1a4b-718f8f251b31
md"""## Higher-dimensions

n: $(@bind n Slider(3:1:20, default=3, show_value=true))

$(@bind go Button("Sample!"))
"""

# ╔═╡ 7b745dfa-6bdc-11eb-37b5-4fa45a647d4b
begin
	go
	
	hdm = zeros(n)
	hdΣ = kernelmatrix(SqExponentialKernel(), reshape(collect(range(-3.0,3.0,length=n)),:,1), obsdim=1)
	hdmvnorm = MultivariateNormal(hdm, hdΣ)
	samp = rand(hdmvnorm, 1)
	plot(collect(1:n), samp, ylims=(-3., 3.), leg=false, linewidth=3)
	scatter!(collect(1:n), samp)
	xlabel!("Variable index")
end

# ╔═╡ 4d1fb5ba-6c79-11eb-25fa-83c5f07286e5
md"""## Gaussian process samples

kernel: $(@bind kernel_op Select(["sqexp" => "RBF", "mat12" => "Matern12", "mat32" => "Matern32", "mat52" => "Matern52"], default="sqexp"))

lengthscale: $(@bind ℓ Slider(0.01:0.01:2.0, default=1., show_value=true))

variance: $(@bind α Slider(0.01:0.01:5.0, default=1., show_value=true))

number of samples: $(@bind n_samples Slider(1:1:20, default=1, show_value=true))


$(@bind go2 Button("Sample!"))
"""

# ╔═╡ ed96aaf8-6c83-11eb-2cb3-0794eea68112
kernel_func = Dict("sqexp" => SqExponentialKernel(), "mat12" => Matern12Kernel(), "mat32" => Matern32Kernel(), "mat52" => Matern52Kernel())[kernel_op]

# ╔═╡ 218d00a8-6c7c-11eb-1f27-97a5e6890d4c
begin
	go2
	
	x = reshape(collect(range(-2., 2., length=100)),:, 1)
	K = kernelmatrix(α*kernel_func, x/ℓ, obsdim=1)
	jitter = I(100)*1e-6
	mvn = MultivariateNormal(zeros(100), K+jitter)
	samples = rand(mvn, n_samples)
	p1 = plot(x, samples, color=:blue, alpha=0.5, leg=false)
	p2 = plot(heatmap(K + jitter, yflip=true, colorbar=false))
	plot(p1, p2)
	plot!(size=(670, 280))
end

# ╔═╡ 4d3a865e-6c87-11eb-16cf-a35a981f9789
md"""## More exotic kernels
"""

# ╔═╡ 5d66a542-6c87-11eb-2518-7fedde0580cc
begin
	 X = reshape(collect(range(-10.0,10.0,length=100)),:,1)
  # Set simple scaling of the data
  k₁ = SqExponentialKernel()
  K₁ = kernelmatrix(k₁,X,obsdim=1)

  # Set a function transformation on the data
  k₂ = TransformedKernel(Matern32Kernel(),FunctionTransform(x->sin.(x)))
  K₂ = kernelmatrix(k₂,X,obsdim=1)

  # Set a matrix premultiplication on the data
  k₃ = PolynomialKernel(c=10.0,d=2)
  K₃ = kernelmatrix(k₃,X,obsdim=1)

  # Add and sum kernels
  k₄ = 0.5*SqExponentialKernel()*LinearKernel(c=0.5) + 0.4*k₂
  K₄ = kernelmatrix(k₄,X,obsdim=1)

  plot(heatmap.([K₁,K₂,K₃,K₄],yflip=true,colorbar=false)...,layout=(2,2),title=["RBF" "Sinusoidal Matern32" "Polynomial" "RBF * Linear"])
end

# ╔═╡ Cell order:
# ╟─bac88c10-6b19-11eb-3368-e180db4a6825
# ╟─3791104a-6b1b-11eb-11d5-8dbda458e0f3
# ╟─ac90924a-6b1a-11eb-2f1c-27fc62ccc345
# ╟─1016fae2-6b1c-11eb-1aa6-71ec793159bd
# ╟─f2b9c820-6b1c-11eb-081a-2b1f6fa548f3
# ╟─4c04d3bc-6bdc-11eb-1a4b-718f8f251b31
# ╟─7b745dfa-6bdc-11eb-37b5-4fa45a647d4b
# ╟─4d1fb5ba-6c79-11eb-25fa-83c5f07286e5
# ╟─ed96aaf8-6c83-11eb-2cb3-0794eea68112
# ╟─218d00a8-6c7c-11eb-1f27-97a5e6890d4c
# ╟─4d3a865e-6c87-11eb-16cf-a35a981f9789
# ╠═5d66a542-6c87-11eb-2518-7fedde0580cc
