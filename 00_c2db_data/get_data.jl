### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ be805156-7b73-42bc-8165-2153870416a8
using JLD2

# ╔═╡ 8536aea2-d305-4421-8fbb-4fc5230e4b43
using PeriodicTable

# ╔═╡ 46c8bbe1-7ed6-4de3-ac40-a3194facdfbe
using LinearAlgebra

# ╔═╡ 189a54a1-9a77-44d2-a44f-283d32b07459
md"""
## 获取所有材料的信息
"""

# ╔═╡ 8579f3ba-4416-46ed-964f-fc6e763a2120
raw_data = load("C2DB.jld2", "data")

# ╔═╡ 6bc4e7c6-f146-4e77-81c2-be6b221c8ce8
function config_produce(x::String)
	if x == "ph_sh"
		return (1, 1)
	elseif x == "ph_sl"
		return (1, 0)
	elseif x == "pl_sh"
		return (0, 1)
	elseif x == "pl_sl"
		return (0, 0)
	else
		error("Wrong config!")
	end
end

# ╔═╡ 5663ac31-00f3-4e8a-852b-49beac3ec493
get_stiff_mat(x; T=Float32) = reshape(
	x["stiffness"]["kwargs"]["data"]["stiffness_tensor"]["__ndarray__"][3],
	(3, :)
) .|> T

# ╔═╡ edf7c6c0-22f3-47e1-8eb5-8aee8eef0d0f
get_pos_mat(x; T=Float32) = reshape(
	x["structure"]["1"]["positions"]["__ndarray__"][3],
	(3, :)
) .|> T

# ╔═╡ 66bf73b3-6b63-4d3b-beb1-5853d49baf26
get_cell_mat(x; T=Float32) = reshape(
	x["structure"]["1"]["cell"]["array"]["__ndarray__"][3],
	(3, :)
) .|> T

# ╔═╡ 64c59b57-b550-4e35-971a-1088af8fb94e
get_numbers(x; T=Int) = x["structure"]["1"]["numbers"]["__ndarray__"][3] .|> T

# ╔═╡ f6156390-a0a6-4b06-9feb-2ff1f35d8630
get_therm_dyn(x; T=Int) = x["hull"]["kwargs"]["data"]["thermodynamic_stability_level"] |> T

# ╔═╡ 726b8e0b-9a8a-48bb-bc57-e46ffbb3939b
raw_data["Co2Cl6-bdec053d68e4"]

# ╔═╡ 150f2f53-bbbe-46a5-8716-f65d80ea1fb4
raw_data["Co2Cl6-bdec053d68e4"] |> get_cell_mat

# ╔═╡ aae212a6-09ee-49c7-a4ec-1a6fd9915320
max_atom = raw_data |> values .|> get_numbers .|> length |> maximum

# ╔═╡ b9ba91de-6b6b-406b-87bd-862f1d47a095
id_vec = raw_data |> keys |> collect

# ╔═╡ 0fc6a8fb-a34f-43d3-9b45-fc968b60be48
material_count = length(id_vec)

# ╔═╡ 02b5be5d-b968-4a84-8daf-762a220c887c
dis_mat = zeros(Float32, max_atom, max_atom, 3, material_count);

# ╔═╡ 6f686437-f3e6-4833-877b-2873be077a49
number_vec = Vector{Vector{Int}}(undef, material_count);

# ╔═╡ 1104c869-e184-48e7-9a97-66df9915e0de
stiffness_mat = zeros(Float32, 3, 3, material_count);

# ╔═╡ 3e9dcbb5-351d-44c7-934d-c8327574de5d
dyn_mat = zeros(Int, 3, material_count);

# ╔═╡ 942db683-e7e9-436c-9ea7-dcbac68d36af
cell_mat_vec = Vector{Matrix{Float32}}(undef, material_count);

# ╔═╡ 6cd649b6-c69a-43ac-b5e7-96980f76783f
pos_mat_vec = Vector{Matrix{Float32}}(undef, material_count);

# ╔═╡ 28e28b73-7742-4df7-ae5e-99660b958e56
for (material_index, material_id) in enumerate(id_vec)
	material = raw_data[material_id]

	cell_mat = get_cell_mat(material)
	pos_mat = get_pos_mat(material)
	cell_mat_vec[material_index] = cell_mat
	pos_mat_vec[material_index] = pos_mat
	atom_count = size(pos_mat, 2)
	for axis_index = 0:2
		if axis_index == 0
			aug_vec = zeros(Float32, 3)
			aug_mat = I
		else
			aug_vec = cell_mat[:, axis_index]
			aug_mat = zeros(Float32, atom_count, atom_count)
		end

		dis_mat[1:atom_count, 1:atom_count, axis_index + 1, material_index] .= [
			norm(@. i_pos - j_pos + aug_vec) for i_pos in eachcol(pos_mat), j_pos in eachcol(pos_mat)
		] + aug_mat
	end

	number_vec[material_index] = get_numbers(material)
	stiffness_mat[:, :, material_index] .= get_stiff_mat(material)
	dyn_mat[1, material_index] = get_therm_dyn(material)
	dyn_mat[2:3, material_index] .= config_produce(material["config"])
end

# ╔═╡ 42ce696a-8295-496f-996f-33d7406469e5
save(
	"data.jld2",
	Dict(
		"id_vec" => id_vec,
		"dis_mat" => dis_mat,
		"number_vec" => number_vec,
		"stiff_mat" => stiffness_mat,
		"dyn_mat" => dyn_mat,
		"cell_mat_vec" => cell_mat_vec,
		"pos_mat_vec" => pos_mat_vec
	)
)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PeriodicTable = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"

[compat]
JLD2 = "~0.4.25"
PeriodicTable = "~1.1.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "41a185d3bf4a06049642523f8b56685ea452d1ae"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "7be5f99f7d15578798f338f5433b6c432ea8037b"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "1c3ff7416cb727ebf4bab0491a56a296d7b8cf1d"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.25"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PeriodicTable]]
deps = ["Base64", "Test", "Unitful"]
git-tree-sha1 = "5ed1e2691eb13b6e955aff1b7eec0b2401df208c"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.1.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Unitful]]
deps = ["ConstructionBase", "Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d57a4ed70b6f9ff1da6719f5f2713706d57e0d66"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.12.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═be805156-7b73-42bc-8165-2153870416a8
# ╠═8536aea2-d305-4421-8fbb-4fc5230e4b43
# ╠═46c8bbe1-7ed6-4de3-ac40-a3194facdfbe
# ╟─189a54a1-9a77-44d2-a44f-283d32b07459
# ╠═8579f3ba-4416-46ed-964f-fc6e763a2120
# ╠═6bc4e7c6-f146-4e77-81c2-be6b221c8ce8
# ╠═5663ac31-00f3-4e8a-852b-49beac3ec493
# ╠═edf7c6c0-22f3-47e1-8eb5-8aee8eef0d0f
# ╠═66bf73b3-6b63-4d3b-beb1-5853d49baf26
# ╠═64c59b57-b550-4e35-971a-1088af8fb94e
# ╠═f6156390-a0a6-4b06-9feb-2ff1f35d8630
# ╠═726b8e0b-9a8a-48bb-bc57-e46ffbb3939b
# ╠═150f2f53-bbbe-46a5-8716-f65d80ea1fb4
# ╠═aae212a6-09ee-49c7-a4ec-1a6fd9915320
# ╠═b9ba91de-6b6b-406b-87bd-862f1d47a095
# ╠═0fc6a8fb-a34f-43d3-9b45-fc968b60be48
# ╠═02b5be5d-b968-4a84-8daf-762a220c887c
# ╠═6f686437-f3e6-4833-877b-2873be077a49
# ╠═1104c869-e184-48e7-9a97-66df9915e0de
# ╠═3e9dcbb5-351d-44c7-934d-c8327574de5d
# ╠═942db683-e7e9-436c-9ea7-dcbac68d36af
# ╠═6cd649b6-c69a-43ac-b5e7-96980f76783f
# ╠═28e28b73-7742-4df7-ae5e-99660b958e56
# ╠═42ce696a-8295-496f-996f-33d7406469e5
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
