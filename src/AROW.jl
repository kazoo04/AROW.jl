module AROW

export fit, predict

type Classifier
  dim::UInt64
  mean::Array{Float64, 1}
  cov::Array{Float64, 1}
  r::Float64

  function Classifier(dimension, param = 1.0)
    new(dimension, rand(dimension), rand(dimension), param)
  end
end

type MultiClassifier
  n_class::UInt64
  arows::Array{Classifier, 1}

  function MultiClassifier(dimension, num_classes, param = 1.0)
    arows = []
    for i=1:num_classes
      push!(arows, Classifier(dimension, param))
    end
    new(num_classes, arows)
  end
end

function fit{T<:AbstractArray}(classifier::MultiClassifier, x::T, label::Int)
  for i=1:classifier.n_class
    if i == label
      fit(classifier.arows[i], x, 1)
    elseif rand() <= 1.0 / (classifier.n_class - 1)
      fit(classifier.arows[i], x, -1)
    end
  end
end

function predict{T<:AbstractArray}(classifier::MultiClassifier, x::T)
  results = []

  for i=1:classifier.n_class
    score = dot(classifier.arows[i].mean, x)
    push!(results, score)
  end

  indmax(results)
end

function fit{T<:AbstractArray}(arow::Classifier, x::T, label::Int)
  assert(label == 1 || label == -1);

  margin = dot(arow.mean, x)

  if margin * label >= 1
    return 0
  end

  confidence = dot(arow.cov, x .* x)
  beta       = 1.0 / (confidence + arow.r)
  alpha      = (1.0 - label * margin) * beta; 

  # update mean
  BLAS.axpy!(alpha * label, arow.cov .* x, arow.mean)

  # update covariance
  @inbounds for i=1:arow.dim
    arow.cov[i] = 1.0 / ((1.0 / arow.cov[i]) + x[i]^2 / arow.r)
  end

  ifelse(margin * label < 0, 1, 0)
end

function predict{T<:AbstractArray}(arow::Classifier, x::T)
  ifelse(dot(arow.mean, x) > 0, 1, -1)
end

end # module
