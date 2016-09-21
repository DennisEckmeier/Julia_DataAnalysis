function fast_tsne(X; no_dims = 2, initial_dims = 50, perplexity = 30, theta=0.5)

  #=
   Julia wrapper for Barnes-hut t-SNE by Dennis Eckmeier, 2016

   t-SNE, the C++ implementation of Barnes-Hut accelerated t-SNE, and the
   Matlab wrapper on which this code is based were developed and provided by
          Laurens van der Maaten (Delft University of Technology)
          https://lvdmaaten.github.io/tsne/
   =#

   #=
          FAST_TSNE Runs the C++ implementation of Barnes-Hut t-SNE

     mappedX = fast_tsne(X, no_dims, initial_dims, perplexity, theta)

   Runs the C++ implementation of Barnes-Hut-SNE. The high-dimensional
   datapoints are specified in the NxD matrix X. The dimensionality of the
   datapoints is reduced to initial_dims dimensions using PCA (default = 50)
   before t-SNE is performed. Next, t-SNE reduces the points to no_dims
   dimensions. The perplexity of the input similarities may be specified
   through the perplexity variable (default = 30). The variable theta sets
   the trade-off parameter between speed and accuracy: theta = 0 corresponds
   to standard, slow t-SNE, while theta = 1 makes very crude approximations.
   Appropriate values for theta are between 0.1 and 0.7 (default = 0.5).
   The function returns the two-dimensional data points in mappedX.

   NOTE: The function is designed to run on large (N > 5000) data sets. It
   may give poor performance on very small data sets (it is better to use a
   standard t-SNE implementation on such data).

   Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
   All rights reserved.


   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   3. All advertising materials mentioning features or use of this software
      must display the following acknowledgement:
      This product includes software developed by the Delft University of Technology.
   4. Neither the name of the Delft University of Technology nor the names of
      its contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
   EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
   IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
   OF SUCH DAMAGE.
  =#

# Perform the initial dimensionality reduction using PCA

  X = X .- mean(X,1)
  covX = X' * X
  (λ, M) = eig(covX) # Python style eig -> (D,V) versus Matlab [V,D] = eig()
  ind = sortperm(λ , rev=true)

  if initial_dims > size(M,2)
    initial_dims = size(M,2)
  end

  M = M[:,ind[1:initial_dims]]
  X = X * M

  # cleaning up memory
  covX = 0
  M = 0
  λ = 0
  gc()

  # Run the fast diffusion SNE implementation
   write_data(X, no_dims, theta, perplexity)
   run(`bh_tsne`)
   (mappedX, landmarks, costs) = read_data();

   rm("data.dat")
   rm("result.dat")

  # ----------------------------------------------------------


  return mappedX

end

# Writes the datafile for the fast t-SNE implementation
function write_data(X, no_dims, theta, perplexity)
  n, d = size(X)
  h = open("data.dat","w")
  A = write(h, Int32(n), Int32(d), Float64(theta), Float64(perplexity), Int32(no_dims))
  # X=X'
  for i in eachindex(X)
      write(h,Float64(X[i]))
  end
    close(h)
end



# Reads the result file from the fast t-SNE implementation
function read_data()
  h = open("result.dat","r")
  n = read(h, Int32)
  d = read(h, Int32)
  X = read(h, Float64,n*d)
  landmarks = read(h, Int32,n)
  costs = read(h, Float64, n)      # this vector contains only zeros
  X = reshape(X, (d, n))'
  close(h)
 return X, landmarks, costs
end
#
