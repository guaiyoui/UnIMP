import torch.optim as optim
import numpy as np
import torch
from collections import Counter

from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine
from Levenshtein import distance as levenshtein_distance
import numpy as np
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from scipy import optimize

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer


def get_main_device(model):
    return next(iter(set(p.device for p in model.parameters() if p.device.type == 'cuda')))


##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params =.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).
    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask

def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    q : float
        Quantile level at which the cuts should occur
    p_params : float
        Proportion of variables that will have missing values
    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.
        
    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.
        
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """
    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1) ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False) ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = torch.quantile(X[:, idxs_na], 1-q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = torch.quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = torch.quantile(X[:, idxs_na], 1-q, dim=0)
        l_quants = torch.quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
    ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, dtype=X.dtype)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            try:
                intercepts[j] = optimize.bisect(f, -50, 50)
            except ValueError as e:
                print(f"Error in bisection method for self_mask at index {j}: {e}")
                intercepts[j] = float('nan')
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            try:
                intercepts[j] = optimize.bisect(f, -50, 50)
            except ValueError as e:
                print(f"Error in bisection method at index {j}: {e}")
                intercepts[j] = float('nan')
    return intercepts


def produce_NA(X, p_miss, mecha="MCAR", n_row=None, n_col=None, opt=None, p_obs=None, q=None):
    """
    Generates a mask for missing values in a dataset based on specified missing data mechanisms.

    Parameters:
    X (torch.Tensor): The input data tensor from which missing values will be generated.
    p_miss (float): The proportion of missing values to introduce.
    mecha (str): The mechanism for generating missing values. Options include:
        - "Random": Randomly assigns missing values.
        - "MCAR": Missing Completely At Random.
        - "MAR": Missing At Random, using the MAR_mask function.
        - "MNAR": Missing Not At Random, with options for quantile or self-masked methods.
    n_row (int, optional): The number of rows in the input data. Required for certain mechanisms.
    n_col (int, optional): The number of columns in the input data. Required for certain mechanisms.
    opt (str, optional): Additional option for MNAR mechanism, specifying the method to use.
    p_obs (float, optional): The proportion of observed values, used in MAR and MNAR mechanisms.
    q (float, optional): A quantile value used in the MNAR mechanism with quantile option.

    Returns:
    torch.Tensor: A tensor representing the mask for missing values, where 1 indicates observed values and 0 indicates missing values.
    
    Raises:
    ValueError: If the specified missing mechanism is not implemented.

    Example:
    >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> mask = produce_NA(X, p_miss=0.2, mecha="MCAR", n_row=2, n_col=2)
    """
    if mecha == "Random":
        unif_random_matrix = np.random.uniform(0., 1., size=X.shape[0])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
    elif mecha == "MCAR":
        unif_random_matrix = np.random.uniform(0., 1., size=[n_row, n_col])
        binary_random_matrix = 1 * (unif_random_matrix < (1 - p_miss))
        mask = torch.FloatTensor(binary_random_matrix) == 1
    elif mecha == "MAR":
        mask = MAR_mask(X.view(n_row, n_col), p_miss, p_obs).double()
        mask = mask == 0
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X.view(n_row, n_col), p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X.view(n_row, n_col), p_miss).double()
    elif mecha == "MNAR" and opt == "logistic":
        if torch.is_tensor(X):
            mask = MNAR_mask_logistic(X.double(), p_miss, p_obs).double()
        else:
            mask = MNAR_mask_logistic(torch.from_numpy(X).double(), p_miss, p_obs).double()
        mask = mask == 0
    else:
        raise ValueError("Missing mechanism not implemented")
    return mask.view(-1)

def skip_bigrams(text, k=2):
    tokens = text.split()
    skip_bigrams_set = set()
    for i in range(len(tokens)):
        for j in range(i + 1, min(i + k + 1, len(tokens))):
            skip_bigrams_set.add((tokens[i], tokens[j]))
    return skip_bigrams_set

def compute_rouge_s(label, pred):
    pred_bigrams = skip_bigrams(pred)
    label_bigrams = skip_bigrams(label)

    intersection = pred_bigrams.intersection(label_bigrams)
    
    precision = len(intersection) / len(pred_bigrams) if pred_bigrams else 0.0
    recall = len(intersection) / len(label_bigrams) if label_bigrams else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def lcs_length(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    
    return L[m][n]

def compute_rouge_w(label, pred):
    pred_tokens = pred.split()
    label_tokens = label.split()

    lcs_len = lcs_length(pred_tokens, label_tokens)
    
    precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs_len / len(label_tokens) if label_tokens else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def jaccard_sim(str1, str2):
    
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    intersection = set1.intersection(set2)

    union = set1.union(set2)

    jaccard_similarity = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    return jaccard_similarity

def cosine_sim_tf(text1, text2):
    
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]

def cosine_sim_tfidf(text1, text2):
    
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]


def cosine_sim_word_embeddings(text1, text2, model, tokenizer):
    # Tokenize and encode the texts
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    # Use the [CLS] token embedding as the sentence representation
    embedding1 = outputs1.last_hidden_state[:, 0, :].numpy()
    embedding2 = outputs2.last_hidden_state[:, 0, :].numpy()
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(embedding1, embedding2)
    return cosine_sim[0][0]


def compute_LLM_generation_metrics(pred_test_all, label_test_all):
    # Initialize scorers
    rouge_scorer_ins = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge = Rouge()

    bleu_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    rouge_lsum_scores = []
    rouge_w_scores = []
    rouge_s_scores = []
    jaccard_sims = []
    lev_distances = []
    cos_sims = []
    cos_sims_tf = []
    cos_sims_tfidf = []
    cos_sims_word_embeddings = []

    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    for preds, labels in zip(pred_test_all, label_test_all):
        for pred, label in zip(preds, labels):
            # Remove "<eos> " prefix if present
            pred = pred.replace(" <eos>", "").strip()
            # Ensure pred is not an empty string
            if not pred:
                pred = "<eos>"
            pred = pred.replace(" Question", "").strip()
            
            label = label[0] if isinstance(label, list) else label

            # print(label, pred)
            label = str(label)
            pred = str(pred)

            chencherry = SmoothingFunction()
            bleu_score = sentence_bleu([label.split()], pred.split(), smoothing_function=chencherry.method1)
            bleu_scores.append(bleu_score)

            # ROUGE Scores
            rouge_scores = rouge_scorer_ins.score(label, pred)
            rouge_1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge_2_scores.append(rouge_scores['rouge2'].fmeasure)
            rouge_l_scores.append(rouge_scores['rougeL'].fmeasure)
            rouge_lsum_scores.append(rouge_scores['rougeLsum'].fmeasure)
            rouge_w_scores.append(compute_rouge_w(label, pred)['f1_score'])
            rouge_s_scores.append(compute_rouge_s(label, pred)['f1_score'])

            # Jaccard Similarity
            # Ensure that both label and pred are lists for Jaccard calculation
            if isinstance(label, str):
                label = [label]
            if isinstance(pred, str):
                pred = [pred]
            
            # Convert labels to binary format for Jaccard calculation
            label_binary = [1 if word in label else 0 for word in pred]
            pred_binary = [1] * len(pred)
            
            jaccard_sims.append(jaccard_sim(label[0], pred[0]))

            # Levenshtein Distance
            lev_distance = levenshtein_distance(label[0], pred[0])  # Use the first element for distance calculation
            lev_distances.append(lev_distance)

            # Cosine Similarity
            label_vec = np.array([1 if char in label[0] else 0 for char in set(label[0] + pred[0])])
            pred_vec = np.array([1 if char in pred[0] else 0 for char in set(label[0] + pred[0])])
            cos_sim = 1 - cosine(label_vec, pred_vec)
            cos_sims.append(cos_sim)

            print(f"label[0]: {label[0]}, pred[0]: {pred[0]}")
            cos_sims_tf.append(cosine_sim_tf(label[0], pred[0]))
            cos_sims_tfidf.append(cosine_sim_tfidf(label[0], pred[0]))
            cos_sims_word_embeddings.append(cosine_sim_word_embeddings(label[0], pred[0], bert_model, tokenizer))

    # Calculate averages
    avg_bleu = np.mean(bleu_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_rouge_lsum = np.mean(rouge_lsum_scores)
    avg_rouge_w = np.mean(rouge_w_scores)
    avg_rouge_s = np.mean(rouge_s_scores)
    avg_jaccard = np.mean(jaccard_sims)
    avg_levenshtein = np.mean(lev_distances)
    avg_cosine = np.mean(cos_sims)
    avg_cosine_tf = np.mean(cos_sims_tf)
    avg_cosine_tfidf = np.mean(cos_sims_tfidf)
    avg_cosine_word_embeddings = np.mean(cos_sims_word_embeddings)

    return avg_bleu, avg_rouge_1, avg_rouge_l, avg_rouge_lsum, avg_rouge_w, avg_rouge_s, avg_jaccard, avg_levenshtein, avg_cosine, avg_cosine_tf, avg_cosine_tfidf, avg_cosine_word_embeddings

    