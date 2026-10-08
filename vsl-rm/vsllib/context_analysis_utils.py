
import os
from typing import Any, Dict, Optional
import re
from collections import Counter
import re
import unicodedata
from importlib import import_module
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from stopwordsiso import langs as iso_languages
from stopwordsiso import stopwords as iso_stopwords
import torch as th
from torch.optim.optimizer import Optimizer as Optimizer
from kNLPmeans.summaryCentroids import build_sentence_corpus, embed_sentences, summarize_textrank

from vsllib.defines import (
    LLM_MODEL_EVAL,
    LLM_PROVIDER_BASE_URL,
    OPENROUTER_FREE_FALLBACK_MODELS,
    GROQ_FREE_FALLBACK_MODELS,
    LLM_PROVIDER_DEFAULT_MODEL,
    LLM_PROVIDER_ENV_VAR,
    ContextImplementations,
    LLMProvider,
)

from vsllib.utils import flatten_metrics_for_csv, format_value_outcome_agreements
def _coherence_and_representativeness_title_lines(cm: Dict[str, float]) -> list:
    """Coherence_{i}/representativeness lines for a wordcloud/bar-plot title, in value
    order (no per-value labels -- just the values, in the same order as the value-system
    weights), and the length-vs-reward Pearson/Spearman correlation lines that go with
    them, if present.
    """
    lines = []
    coherence_keys = sorted(
        (k for k in cm if re.fullmatch(r"coherence_\d+", k)),
        key=lambda k: int(k.split("_")[1]))
    if coherence_keys:
        lines.append("coherence: [" + ", ".join(f"{cm[k]:.3f}" for k in coherence_keys) + "]")
    if "representativeness" in cm:
        lines.append(f"representativeness: {cm['representativeness']:.3f}")

    for label, key_prefix in (
        ("GTGR-To-GTVS", "gtgr_to_gtvs_agreement"),
        ("LRGR-To-GTVS", "lrgr_to_gtvs_agreement"),
        ("LRGR-To-LRVS", "lrgr_to_lrvs_agreement"),
    ):
        outcome_agreement_keys = sorted(
            (k for k in cm if re.fullmatch(rf"{key_prefix}_\d+", k)),
            key=lambda k: int(k.rsplit("_", 1)[-1]))
        if outcome_agreement_keys:
            lines.append(f"{label}: [" + ", ".join(f"{cm[k]:.1f}%" for k in outcome_agreement_keys) + "]")

    pearson_keys = sorted(
        (k for k in cm if re.fullmatch(r"length_pearson_\d+", k)),
        key=lambda k: int(k.rsplit("_", 1)[-1]))
    spearman_keys = sorted(
        (k for k in cm if re.fullmatch(r"length_spearman_\d+", k)),
        key=lambda k: int(k.rsplit("_", 1)[-1]))
    if pearson_keys:
        lines.append("length-reward pearson: [" + ", ".join(f"{cm[k]:.3f}" for k in pearson_keys) + "]")
    if spearman_keys:
        lines.append("length-reward spearman: [" + ", ".join(f"{cm[k]:.3f}" for k in spearman_keys) + "]")
    if "length_pearson_vs" in cm or "length_spearman_vs" in cm:
        pr_vs = cm.get("length_pearson_vs", float("nan"))
        sr_vs = cm.get("length_spearman_vs", float("nan"))
        lines.append(f"length-VS reward pearson: {pr_vs:.3f}, spearman: {sr_vs:.3f}")
    return lines


_WORD_CLOUD_STOPWORDS = frozenset(
    word.casefold()
    for language in iso_languages()
    for word in iso_stopwords(language)
)
_WORD_CLOUD_TOKEN_PATTERN = re.compile(r"(?u)\b[\w'-]+\b")


def _multilingual_word_frequencies(text: str) -> Dict[str, int]:
    """Tokenize text without making a language assumption."""
    frequencies = Counter()
    for token in _WORD_CLOUD_TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold()):
        token = token.strip("'_-")
        if len(token) > 1 and token not in _WORD_CLOUD_STOPWORDS and not token.isnumeric():
            frequencies[token] += 1
    return dict(frequencies)



def plot_cluster_word_clouds(texts, labels, output_path: str, clustering_name: str, vs_predicted=None, label_names=None, descriptions=None, cluster_metrics=None, value_names=None, value_outcome_agreements=None, use_llm_categories: bool = False) -> None:
        texts = np.asarray(texts, dtype=object)
        if isinstance(labels, th.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and {clustering_name} labels must have the same length, "
                f"got {len(texts)} and {len(labels)}"
            )
        if vs_predicted is not None:
            if isinstance(vs_predicted, th.Tensor):
                vs_predicted = vs_predicted.detach().cpu().numpy()
            vs_predicted = np.asarray(vs_predicted)
            if len(vs_predicted) != len(labels):
                raise ValueError(
                    f"vs_predicted and {clustering_name} labels must have the same length, "
                    f"got {len(vs_predicted)} and {len(labels)}"
                )

        panels = []
        for cluster_label in np.unique(labels):
            cluster_mask = labels == cluster_label
            cluster_texts = [str(text) for text in texts[cluster_mask] if str(text).strip()]
            if cluster_texts:
                average_vs = None
                if vs_predicted is not None:
                    average_vs = np.mean(vs_predicted[cluster_mask], axis=0)
                panels.append((cluster_label, len(cluster_texts), " ".join(cluster_texts), average_vs))

        panels.sort(key=lambda panel: -panel[1])

        if not panels:
            return

        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 4.5 * rows),
            squeeze=False,
        )
        for axis, (cluster_label, cluster_size, text, average_vs) in zip(axes.flat, panels):
            word_frequencies = _multilingual_word_frequencies(text)
            if word_frequencies:
                word_cloud = WordCloud(
                    width=800,
                    height=600,
                    background_color="white",
                    random_state=0,
                ).generate_from_frequencies(word_frequencies)
                axis.imshow(word_cloud, interpolation="bilinear")
                popular_words = list(word_cloud.words_)[:4]
            else:
                popular_words = []
                axis.text(
                    0.5,
                    0.5,
                    "No non-stopword tokens",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.axis("off")
            title_words = ", ".join(popular_words) or f"Cluster {cluster_label}"
            if use_llm_categories and label_names is not None:
                title_words = label_names.get(cluster_label, title_words)
            title = f"{clustering_name}, {title_words} (n={cluster_size})"
            if average_vs is not None:
                average_vs_text = ", ".join(f"{weight:.3f}" for weight in np.ravel(average_vs))
                title += f"\nmean predicted VS: [{average_vs_text}]"
            if cluster_metrics is not None and cluster_label in cluster_metrics:
                for line in _coherence_and_representativeness_title_lines(cluster_metrics[cluster_label]):
                    title += f"\n{line}"
            """THIS IS TOO MUCH INFO... if descriptions is not None and cluster_label in descriptions:
                title += f"\n{descriptions[cluster_label]}"
            """
            axis.set_title(title)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        suptitle = format_value_outcome_agreements(value_outcome_agreements)
        figure.tight_layout(pad=2.5, h_pad=4.0, w_pad=3.0)
        figure.subplots_adjust(hspace=0.65, wspace=0.35, top=0.86 if suptitle else None)
        if suptitle is not None:
            figure.suptitle(suptitle, fontsize=11, y=0.99)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)

def plot_cluster_metrics_bars(cluster_metrics: Dict[Any, Dict[str, float]], value_names, output_path: str, clustering_name: str, value_outcome_agreements=None) -> None:
        """Bar-plot grid (one subplot per cluster) of per-value grounding accuracy
        (`coherence_{i}`) and value-system accuracy (`representativeness`) for a single
        clustering (as produced by `compute_metrics_per_cluster`).
        """
        value_names = list(value_names)
        panels = sorted(cluster_metrics.items(), key=lambda kv: -kv[1].get("size", 0))
        if not panels:
            return

        bar_labels = value_names + ["representativeness"]
        colors = ["tab:blue"] * len(value_names) + ["tab:orange"]

        columns = min(4, len(panels))
        rows = (len(panels) + columns - 1) // columns
        figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 5.5 * rows), squeeze=False)
        for axis, (cluster_label, metrics) in zip(axes.flat, panels):
            values = [metrics.get(f"coherence_{i}", np.nan) for i in range(len(value_names))] + [metrics.get("representativeness", np.nan)]
            bars = axis.bar(range(len(bar_labels)), values, color=colors)
            axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=2, fontsize=8)
            axis.set_xticks(range(len(bar_labels)))
            axis.set_xticklabels(bar_labels, rotation=45, ha="right")
            axis.set_ylim(0, 1.08)
            axis.set_ylabel("accuracy")
            label_text = f"Cluster {cluster_label}" if isinstance(cluster_label, (int, np.integer)) else str(cluster_label)
            title = f"{clustering_name}, {label_text} (n={metrics.get('size', 0)})"
            for line in _coherence_and_representativeness_title_lines(metrics):
                if not line.startswith("coherence:") and not line.startswith("representativeness:"):
                    title += f"\n{line}"
            axis.set_title(title)
        for axis in axes.flat[len(panels):]:
            axis.axis("off")
        suptitle = format_value_outcome_agreements(value_outcome_agreements)
        figure.tight_layout(pad=2.5, h_pad=5.0, w_pad=3.0)
        figure.subplots_adjust(hspace=1.0, wspace=0.4, top=0.86 if suptitle else None)
        if suptitle is not None:
            figure.suptitle(suptitle, fontsize=11, y=0.99)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)

def save_cluster_metrics(cluster_metrics_by_clustering: Dict[str, Dict[Any, Dict[str, float]]], output_dir: str) -> pd.DataFrame:
        """Writes every accuracy metric from `compute_metrics_per_cluster`, for every
        clustering, to `{output_dir}/cluster_metrics.csv` and `.json` -- one row per
        (clustering, cluster) pair.
        """
        rows = []
        for clustering_name, per_cluster in cluster_metrics_by_clustering.items():
            for cluster_label, metrics in per_cluster.items():
                row = {"clustering": clustering_name, "cluster": cluster_label}
                row.update(flatten_metrics_for_csv(metrics))
                rows.append(row)
        descriptions = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        descriptions.to_csv(os.path.join(output_dir, "cluster_metrics.csv"), index=False)
        descriptions.to_json(os.path.join(output_dir, "cluster_metrics.json"), orient="records", indent=2)
        return descriptions

def describe_clusters_with_llm(texts, label_sets, clustering_names, output_dir: str, model_name: Optional[str] = None, provider: str = LLMProvider.GROQ.value, max_documents: int = 20, cluster_metrics: Optional[Dict[str, Dict[Any, Dict[str, float]]]] = None):
        """Generate category and description metadata for each text clustering."""
        provider = LLMProvider(provider)
        model_name = model_name or LLM_PROVIDER_DEFAULT_MODEL[provider]
        api_key = os.environ.get(LLM_PROVIDER_ENV_VAR[provider])
        if not api_key:
            raise RuntimeError(f"{LLM_PROVIDER_ENV_VAR[provider]} is not set; cannot generate cluster descriptions.")

        if provider == LLMProvider.GROQ:
            try:
                ChatGroq = import_module("langchain_groq").ChatGroq
            except ImportError as error:
                raise ImportError("Install langchain-groq to generate cluster descriptions with Groq.") from error
            make_llm = lambda name: ChatGroq(model=name, temperature=0, api_key=api_key)
            llm = make_llm(model_name).with_fallbacks([make_llm(name) for name in GROQ_FREE_FALLBACK_MODELS if name != model_name])
        elif provider == LLMProvider.OPENROUTER:
            try:
                ChatOpenAI = import_module("langchain_openai").ChatOpenAI
            except ImportError as error:
                raise ImportError("Install langchain-openai to generate cluster descriptions with OpenRouter.") from error
            make_llm = lambda name: ChatOpenAI(model=name, temperature=0, api_key=api_key, base_url=LLM_PROVIDER_BASE_URL[provider])
            llm = make_llm(model_name).with_fallbacks([make_llm(name) for name in OPENROUTER_FREE_FALLBACK_MODELS if name != model_name])
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        texts = np.asarray(texts, dtype=object)
        rows = []
        category_maps = []
        for labels, clustering_name in zip(label_sets, clustering_names):
            labels = np.asarray(labels)
            category_map = {}
            for cluster_label in sorted(np.unique(labels), key=lambda label: (-np.sum(labels == label), str(label))):
                cluster_texts = [str(text) for text in texts[labels == cluster_label] if str(text).strip()]
                sample = cluster_texts[:max_documents]
                if not sample:
                    category = f"Cluster {cluster_label}"
                    description = "No text was available for this cluster."
                    used_model = None
                else:
                    prompt = (
                        "Analyze the following documents from one cluster.\n"
                        "Return exactly two lines:\n"
                        "CATEGORY: a concise descriptive label of at most six words\n"
                        "DESCRIPTION: one concise sentence describing the common themes\n\n"
                        + "\n---\n".join(sample)
                    )
                    response = llm.invoke(prompt)
                    response_text = getattr(response, "content", str(response)).strip()
                    metadata = getattr(response, "response_metadata", None) or {}
                    used_model = metadata.get("model_name") or metadata.get("model") or model_name
                    parsed = {}
                    for line in response_text.splitlines():
                        key, separator, value = line.partition(":")
                        if separator:
                            parsed[key.strip().upper()] = value.strip()
                    category = parsed.get("CATEGORY", response_text.splitlines()[0]).strip()
                    description = parsed.get("DESCRIPTION", response_text).strip()

                category_map[cluster_label] = category
                textrank_summary = summarize_cluster_with_textrank(cluster_texts)

                row = {
                    "clustering": clustering_name,
                    "cluster": cluster_label,
                    "category": category,
                    "description": description,
                    "summary_textrank": textrank_summary,
                    "llm_model": used_model,
                    "size": len(cluster_texts),
                }
                cm = (cluster_metrics or {}).get(clustering_name, {}).get(cluster_label, {})
                for key, value in cm.items():
                    if key == "representativeness" or re.fullmatch(
                        r"coherence_\d+|gtgr_to_gtvs_agreement_\d+|lrgr_to_gtvs_agreement_\d+|lrgr_to_lrvs_agreement_\d+", key
                    ):
                        row[key] = value
                rows.append(row)
            category_maps.append(category_map)

        descriptions = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        descriptions.to_csv(os.path.join(output_dir, "cluster_descriptions.csv"), index=False)
        descriptions.to_json(os.path.join(output_dir, "cluster_descriptions.json"), orient="records", indent=2)
        return descriptions, category_maps

def summarize_cluster_with_textrank(cluster_texts, top_k: int = 5, emb_type: str = "all-MiniLM-L6-v2") -> str:
        if not cluster_texts:
            return "No text was available for this cluster."

        sentences, _ = build_sentence_corpus(cluster_texts)
        if not sentences:
            return "No sentence was available for this cluster."

        sentence_embeddings, _ = embed_sentences(sentences, emb_type=emb_type)
        summary = summarize_textrank(sentences, sentence_embeddings, top_k=top_k)
        return " ".join(sentence for sentence, _ in summary)