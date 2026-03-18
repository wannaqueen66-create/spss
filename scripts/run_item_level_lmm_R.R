#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(jsonlite)
})

sig_tier <- function(p) {
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  ""
}

fmt_num <- function(x, digits = 4) {
  ifelse(is.na(x), NA_character_, format(round(x, digits), nsmall = digits, trim = TRUE))
}

is_singular_safe <- function(fit) {
  out <- try(lme4::isSingular(fit, tol = 1e-4), silent = TRUE)
  if (inherits(out, "try-error")) return(NA)
  isTRUE(out)
}

collect_warning_flags <- function(warnings_text) {
  wt <- paste(warnings_text, collapse = " | ")
  list(
    warning_text = wt,
    has_warning = nzchar(wt),
    has_convergence_warning = grepl("convergen|failed to converge|negative eigenvalue|unable to evaluate scaled gradient|bobyqa", wt, ignore.case = TRUE),
    has_singularity_warning = grepl("singular", wt, ignore.case = TRUE),
    has_kroger_warning = grepl("kenward-roger|pbkrtest", wt, ignore.case = TRUE)
  )
}

extract_random_effects <- function(fit, dv) {
  vc <- as.data.frame(VarCorr(fit))
  if (nrow(vc) == 0) return(data.frame())
  vc %>%
    transmute(
      DV = dv,
      grp = .data$grp,
      var1 = .data$var1,
      var2 = if ("var2" %in% names(vc)) .data$var2 else NA_character_,
      vcov = .data$vcov,
      sdcor = .data$sdcor
    )
}

extract_fixed_effects <- function(fit, dv, df_method) {
  sm <- summary(fit, ddf = df_method)
  cf <- as.data.frame(sm$coefficients)
  cf$Term <- rownames(cf)
  rownames(cf) <- NULL
  names(cf) <- make.names(names(cf))

  ci <- try(confint(fit, method = "Wald"), silent = TRUE)
  ci_df <- data.frame(Term = character(), CI_low = numeric(), CI_high = numeric())
  if (!inherits(ci, "try-error")) {
    ci_df <- as.data.frame(ci)
    ci_df$Term <- rownames(ci_df)
    rownames(ci_df) <- NULL
    names(ci_df) <- c("CI_low", "CI_high", "Term")
  }

  pcol <- grep("^Pr", names(cf), value = TRUE)
  if (length(pcol) == 0) stop("No p column found in summary coefficients")

  cf %>%
    transmute(
      DV = dv,
      Term = .data$Term,
      Estimate = .data$Estimate,
      SE = .data$Std..Error,
      df = if ("df" %in% names(cf)) .data$df else NA_real_,
      statistic = if ("t.value" %in% names(cf)) .data$t.value else if ("z.value" %in% names(cf)) .data$z.value else NA_real_,
      p = .data[[pcol[1]]]
    ) %>%
    left_join(ci_df, by = "Term") %>%
    mutate(Sig = vapply(.data$p, sig_tier, character(1)))
}

extract_type3 <- function(fit, dv, formula_txt, df_method, requested_re, used_re, fit_method, n_rows, n_subjects) {
  a3 <- anova(fit, type = 3, ddf = df_method)
  a3_df <- as.data.frame(a3)
  a3_df$Effect <- rownames(a3_df)
  rownames(a3_df) <- NULL
  names(a3_df) <- make.names(names(a3_df))
  fcol <- intersect(c("F.value", "Chisq"), names(a3_df))
  pcol <- grep("^Pr", names(a3_df), value = TRUE)
  num_df_col <- intersect(c("NumDF", "npar"), names(a3_df))
  den_df_col <- intersect(c("DenDF", "ddf"), names(a3_df))
  ss_col <- intersect(c("Sum.Sq", "Sum.Sq.", "SSn"), names(a3_df))
  ms_col <- intersect(c("Mean.Sq", "Mean.Sq."), names(a3_df))

  out <- a3_df %>%
    transmute(
      DV = dv,
      Formula = formula_txt,
      Effect = .data$Effect,
      NumDF = if (length(num_df_col) > 0) .data[[num_df_col[1]]] else NA_real_,
      DenDF = if (length(den_df_col) > 0) .data[[den_df_col[1]]] else NA_real_,
      F_value = if (length(fcol) > 0) .data[[fcol[1]]] else NA_real_,
      p = if (length(pcol) > 0) .data[[pcol[1]]] else NA_real_,
      SumSq = if (length(ss_col) > 0) .data[[ss_col[1]]] else NA_real_,
      MeanSq = if (length(ms_col) > 0) .data[[ms_col[1]]] else NA_real_,
      requested_random = requested_re,
      used_random = used_re,
      fit_method = fit_method,
      n_rows = n_rows,
      n_subjects = n_subjects
    ) %>%
    mutate(Sig = vapply(.data$p, sig_tier, character(1)))
  out
}

fit_with_fallback <- function(formula_txt, dat) {
  attempts <- list(
    list(method = "bobyqa", re = "(1 + Complexity | SubjectID)"),
    list(method = "Nelder_Mead", re = "(1 + Complexity | SubjectID)"),
    list(method = "bobyqa", re = "(1 | SubjectID)"),
    list(method = "Nelder_Mead", re = "(1 | SubjectID)")
  )
  last_err <- NULL
  last_warn <- character()
  for (a in attempts) {
    full_formula <- paste0(formula_txt, " + ", a$re)
    warn_log <- character()
    fit <- withCallingHandlers(
      try(
        lmer(
          as.formula(full_formula),
          data = dat,
          REML = FALSE,
          control = lmerControl(optimizer = a$method, calc.derivs = FALSE)
        ),
        silent = TRUE
      ),
      warning = function(w) {
        warn_log <<- c(warn_log, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    if (!inherits(fit, "try-error")) {
      return(list(fit = fit, formula = full_formula, re = a$re, method = a$method, warnings = unique(warn_log)))
    }
    last_err <- as.character(fit)
    last_warn <- unique(c(last_warn, warn_log))
  }
  stop(paste(c(last_err, last_warn), collapse = " | "))
}

normalize_pairwise <- function(df, spec_label) {
  if (nrow(df) == 0) return(df)
  out <- df
  if (!("contrast" %in% names(out))) out$contrast <- NA_character_
  if (!("estimate" %in% names(out))) out$estimate <- NA_real_
  if (!("SE" %in% names(out))) out$SE <- NA_real_
  if (!("df" %in% names(out))) out$df <- NA_real_
  if (!("lower.CL" %in% names(out))) out$lower.CL <- NA_real_
  if (!("upper.CL" %in% names(out))) out$upper.CL <- NA_real_
  if (!("p.value" %in% names(out))) out$p.value <- NA_real_
  out$comparison_direction <- ifelse(is.na(out$estimate), NA_character_, ifelse(out$estimate > 0, "first > second", ifelse(out$estimate < 0, "first < second", "no difference")))
  out$Spec <- spec_label
  out$Sig <- vapply(out$p.value, sig_tier, character(1))
  out
}

safe_emm_summary <- function(obj, infer = c(TRUE, TRUE), adjust = NULL) {
  warn_log <- character()
  val <- withCallingHandlers(
    {
      if (is.null(adjust)) {
        as.data.frame(summary(obj, infer = infer))
      } else {
        as.data.frame(summary(obj, infer = infer, adjust = adjust))
      }
    },
    warning = function(w) {
      warn_log <<- c(warn_log, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(data = val, warnings = unique(warn_log))
}

make_emmeans_outputs <- function(fit, dv, sig_effects, p_adjust) {
  emm_rows <- list()
  pair_rows <- list()
  warn_rows <- list()

  add_emm_pair <- function(spec_label, emm_obj, pair_obj) {
    emm_res <- safe_emm_summary(emm_obj, infer = c(TRUE, TRUE), adjust = NULL)
    emm_df <- emm_res$data
    emm_df$DV <- dv
    emm_df$Spec <- spec_label
    emm_rows[[length(emm_rows) + 1]] <<- emm_df
    if (length(emm_res$warnings) > 0) {
      warn_rows[[length(warn_rows) + 1]] <<- data.frame(DV = dv, Source = paste0("EMM:", spec_label), Warning = emm_res$warnings)
    }

    pair_res <- safe_emm_summary(pair_obj, infer = c(TRUE, TRUE), adjust = p_adjust)
    pair_df <- pair_res$data
    pair_df$DV <- dv
    pair_df <- normalize_pairwise(pair_df, spec_label)
    pair_rows[[length(pair_rows) + 1]] <<- pair_df
    if (length(pair_res$warnings) > 0) {
      warn_rows[[length(warn_rows) + 1]] <<- data.frame(DV = dv, Source = paste0("Pairwise:", spec_label), Warning = pair_res$warnings)
    }
  }

  if ("WWR" %in% sig_effects) {
    emm <- emmeans(fit, ~ WWR)
    add_emm_pair("WWR_main", emm, pairs(emm))
  }
  if ("Complexity" %in% sig_effects) {
    emm <- emmeans(fit, ~ Complexity)
    add_emm_pair("Complexity_main", emm, pairs(emm))
  }
  if ("ExperienceGroup" %in% sig_effects) {
    emm <- emmeans(fit, ~ ExperienceGroup)
    add_emm_pair("ExperienceGroup_main", emm, pairs(emm))
  }
  if ("WWR:Complexity" %in% sig_effects) {
    emm1 <- emmeans(fit, ~ WWR | Complexity)
    add_emm_pair("WWR_within_Complexity", emm1, pairs(emm1))
    emm2 <- emmeans(fit, ~ Complexity | WWR)
    add_emm_pair("Complexity_within_WWR", emm2, pairs(emm2))
  }
  if ("WWR:ExperienceGroup" %in% sig_effects) {
    emm1 <- emmeans(fit, ~ WWR | ExperienceGroup)
    add_emm_pair("WWR_within_ExperienceGroup", emm1, pairs(emm1))
    emm2 <- emmeans(fit, ~ ExperienceGroup | WWR)
    add_emm_pair("ExperienceGroup_within_WWR", emm2, pairs(emm2))
  }
  if ("Complexity:ExperienceGroup" %in% sig_effects) {
    emm1 <- emmeans(fit, ~ Complexity | ExperienceGroup)
    add_emm_pair("Complexity_within_ExperienceGroup", emm1, pairs(emm1))
    emm2 <- emmeans(fit, ~ ExperienceGroup | Complexity)
    add_emm_pair("ExperienceGroup_within_Complexity", emm2, pairs(emm2))
  }
  if ("WWR:Complexity:ExperienceGroup" %in% sig_effects) {
    emm1 <- emmeans(fit, ~ WWR | Complexity * ExperienceGroup)
    add_emm_pair("WWR_within_Complexity_by_ExperienceGroup", emm1, pairs(emm1))
    emm2 <- emmeans(fit, ~ Complexity | WWR * ExperienceGroup)
    add_emm_pair("Complexity_within_WWR_by_ExperienceGroup", emm2, pairs(emm2))
    emm3 <- emmeans(fit, ~ ExperienceGroup | WWR * Complexity)
    add_emm_pair("ExperienceGroup_within_WWR_by_Complexity", emm3, pairs(emm3))
  }

  list(
    emmeans = bind_rows(emm_rows),
    pairwise = bind_rows(pair_rows),
    warnings = bind_rows(warn_rows)
  )
}

effect_label_zh <- function(effect_name) {
  dplyr::case_when(
    effect_name == "WWR" ~ "WWR 主效应",
    effect_name == "Complexity" ~ "Complexity 主效应",
    effect_name == "ExperienceGroup" ~ "ExperienceGroup 主效应",
    effect_name == "WWR:Complexity" ~ "WWR × Complexity 二阶交互",
    effect_name == "WWR:ExperienceGroup" ~ "WWR × ExperienceGroup 二阶交互",
    effect_name == "Complexity:ExperienceGroup" ~ "Complexity × ExperienceGroup 二阶交互",
    effect_name == "WWR:Complexity:ExperienceGroup" ~ "WWR × Complexity × ExperienceGroup 三阶交互",
    TRUE ~ effect_name
  )
}

pick_followup_pairs <- function(effect_name, sub_pair) {
  if (nrow(sub_pair) == 0) return(sub_pair[0, ])
  if (effect_name == "WWR") {
    return(sub_pair %>% filter(.data$Spec == "WWR_main"))
  } else if (effect_name == "Complexity") {
    return(sub_pair %>% filter(.data$Spec == "Complexity_main"))
  } else if (effect_name == "ExperienceGroup") {
    return(sub_pair %>% filter(.data$Spec == "ExperienceGroup_main"))
  } else if (effect_name == "WWR:Complexity") {
    return(sub_pair %>% filter(.data$Spec %in% c("WWR_within_Complexity", "Complexity_within_WWR")))
  } else if (effect_name == "WWR:ExperienceGroup") {
    return(sub_pair %>% filter(.data$Spec %in% c("WWR_within_ExperienceGroup", "ExperienceGroup_within_WWR")))
  } else if (effect_name == "Complexity:ExperienceGroup") {
    return(sub_pair %>% filter(.data$Spec %in% c("Complexity_within_ExperienceGroup", "ExperienceGroup_within_Complexity")))
  } else if (effect_name == "WWR:Complexity:ExperienceGroup") {
    return(sub_pair %>% filter(.data$Spec %in% c("WWR_within_Complexity_by_ExperienceGroup", "Complexity_within_WWR_by_ExperienceGroup", "ExperienceGroup_within_WWR_by_Complexity")))
  }
  sub_pair[0, ]
}

sentence_for_effect <- function(dv, effect_row, pair_df) {
  effect_name <- effect_row$Effect
  effect_label <- effect_label_zh(effect_name)
  base <- sprintf(
    "%s 显著（F(%s, %s) = %s, p = %s, FDR p = %s）",
    effect_label,
    fmt_num(effect_row$NumDF, 2),
    fmt_num(effect_row$DenDF, 2),
    fmt_num(effect_row$F_value, 3),
    fmt_num(effect_row$p, 4),
    fmt_num(effect_row$p_fdr, 4)
  )

  sub_pair <- pair_df %>% filter(.data$DV == dv, !is.na(.data$p.value), .data$p.value < 0.05)
  use <- pick_followup_pairs(effect_name, sub_pair)
  if (nrow(use) == 0) {
    return(paste0(base, "。"))
  }

  top_use <- use %>% arrange(.data$p.value) %>% head(2)
  pair_txt <- vapply(seq_len(nrow(top_use)), function(i) {
    rr <- top_use[i, ]
    sprintf(
      "%s（estimate = %s, 95%% CI [%s, %s], p = %s, %s）",
      rr$contrast,
      fmt_num(rr$estimate, 3),
      fmt_num(rr$lower.CL, 3),
      fmt_num(rr$upper.CL, 3),
      fmt_num(rr$p.value, 4),
      rr$comparison_direction
    )
  }, character(1))

  if (effect_name %in% c("WWR", "Complexity", "ExperienceGroup")) {
    paste0(base, "；pairwise comparisons 显示主要差异集中在：", paste(pair_txt, collapse = "；"), "。")
  } else if (effect_name %in% c("WWR:Complexity", "WWR:ExperienceGroup", "Complexity:ExperienceGroup")) {
    paste0(base, "；simple-effects / pairwise follow-up 提示交互主要体现为：", paste(pair_txt, collapse = "；"), "。")
  } else if (effect_name == "WWR:Complexity:ExperienceGroup") {
    paste0(base, "；三阶交互的 follow-up 比较表明差异主要出现在：", paste(pair_txt, collapse = "；"), "。")
  } else {
    paste0(base, "；follow-up 结果包括：", paste(pair_txt, collapse = "；"), "。")
  }
}

build_narrative <- function(dv, effect_rows, pair_df, caution_df, type3_all_df = NULL) {
  lines <- c()
  all_rows <- if (!is.null(type3_all_df)) type3_all_df %>% filter(.data$DV == dv, !is.na(.data$p_fdr)) else effect_rows

  if (nrow(effect_rows) == 0) {
    if (!is.null(type3_all_df) && nrow(all_rows) > 0 && any(all_rows$p < 0.05, na.rm = TRUE)) {
      raw_only <- all_rows %>% filter(.data$p < 0.05, is.na(.data$p_fdr) | .data$p_fdr >= 0.05)
      if (nrow(raw_only) > 0) {
        lines <- c(lines, sprintf("- %s：有些效应在未校正水平达到 p<.05，但在 FDR 校正后未保留，因此正文建议按未通过多重校正处理。", dv))
      } else {
        lines <- c(lines, sprintf("- %s：Type III fixed effects 中未见显著项（至少在当前 FDR 口径下）。", dv))
      }
    } else {
      lines <- c(lines, sprintf("- %s：Type III fixed effects 中未见显著项（至少在当前 FDR 口径下）。", dv))
    }
  } else {
    sig_sentences <- vapply(seq_len(nrow(effect_rows)), function(i) sentence_for_effect(dv, effect_rows[i, ], pair_df), character(1))
    lead <- sprintf("- %s：在统一结构 LMM 下，%s", dv, paste(sig_sentences, collapse = "；同时，"))
    lines <- c(lines, paste0(lead))

    if (!is.null(type3_all_df) && nrow(all_rows) > 0) {
      raw_only <- all_rows %>% filter(.data$p < 0.05, is.na(.data$p_fdr) | .data$p_fdr >= 0.05)
      if (nrow(raw_only) > 0) {
        lines <- c(lines, sprintf("  - FDR 补充说明：%s 另有未校正显著但经 FDR 后未保留的效应，写作时宜作为趋势或探索性结果处理。", paste(unique(effect_label_zh(raw_only$Effect)), collapse = "、")))
      }
    }
  }

  caut <- caution_df %>% filter(.data$DV == dv)
  if (nrow(caut) > 0) {
    lines <- c(lines, sprintf("  - 谨慎解释：该因变量存在模型 warning（%s），建议结合 fit indices / random effects 一并审阅。", paste(unique(caut$warning_class), collapse = ", ")))
  }
  lines
}

write_report_zh <- function(out_path, summary_df, fdr_df, pair_df, caution_df) {
  lines <- c(
    "# 逐题 / 逐维度 LMM 结果汇总（自动生成）",
    "",
    "固定效应统一为：WWR、Complexity、ExperienceGroup，以及所有二阶 / 三阶交互；随机部分默认尝试 `(1 + Complexity | SubjectID)`，失败时回退到 `(1 | SubjectID)`。",
    "估计方法：ML；Type III fixed effects 使用 lmerTest；显著项 follow-up 使用 estimated marginal means + pairwise comparisons。",
    "多重检验控制：当前汇总文件同时保留原始 p 与 FDR 校正后的 p。正式写作优先参考 FDR 结果。",
    "",
    "## 样本与层级结构说明",
    "- 层级单位：重复测量嵌套于 SubjectID。",
    "- 重复测量因子：WWR、Complexity；被试间分组：ExperienceGroup。",
    "- 每个因变量都使用相同固定效应结构，以便逐题回答哪个主效应或交互影响了哪些题目。",
    ""
  )

  if (nrow(summary_df) > 0) {
    lines <- c(lines, "## 每个因变量的模型状态", "")
    for (i in seq_len(nrow(summary_df))) {
      r <- summary_df[i, ]
      caution_txt <- ifelse(isTRUE(r$caution_flag), sprintf("；caution=%s", r$caution_reason), "")
      lines <- c(lines, sprintf(
        "- %s：status=%s；subjects=%s；rows=%s；random=%s；AIC=%s；BIC=%s；-2LL=%s%s。",
        r$DV, r$Status, r$n_subjects, r$n_rows, r$used_random,
        fmt_num(r$AIC), fmt_num(r$BIC), fmt_num(r$minus2LL), caution_txt
      ))
    }
    lines <- c(lines, "")
  }

  if (nrow(caution_df) > 0) {
    lines <- c(lines, "## 需要谨慎解释的因变量", "")
    for (dv in unique(caution_df$DV)) {
      sub <- caution_df %>% filter(.data$DV == dv)
      lines <- c(lines, sprintf("- %s：%s", dv, paste(unique(sub$warning_class), collapse = ", ")))
    }
    lines <- c(lines, "")
  }

  if (nrow(fdr_df) > 0) {
    lines <- c(lines, "## 按因变量汇总显著结果", "")
    for (dv in unique(fdr_df$DV)) {
      dv_rows <- fdr_df %>% filter(.data$DV == dv, !is.na(.data$p_fdr), .data$p_fdr < 0.05)
      lines <- c(lines, sprintf("### %s", dv))
      lines <- c(lines, build_narrative(dv, dv_rows, pair_df, caution_df, fdr_df), "")
    }
  }

  writeLines(lines, out_path)
}

option_list <- list(
  make_option(c("--long-csv"), type = "character"),
  make_option(c("--out-dir"), type = "character", default = "results/significance/item_level_lmm"),
  make_option(c("--exclude-subjects"), type = "character", default = ""),
  make_option(c("--p-adjust"), type = "character", default = "fdr"),
  make_option(c("--df-method"), type = "character", default = "Satterthwaite")
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$`long-csv`)) stop("--long-csv is required")

out_dir <- opt$`out-dir`
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
for (subdir in c("csv", "md", "json")) dir.create(file.path(out_dir, subdir), recursive = TRUE, showWarnings = FALSE)

x <- read_csv(opt$`long-csv`, show_col_types = FALSE)
if (nzchar(opt$`exclude-subjects`) && "SubjectID" %in% names(x)) {
  ids <- trimws(unlist(strsplit(as.character(opt$`exclude-subjects`), ",", fixed = TRUE)))
  ids <- ids[nzchar(ids)]
  if (length(ids) > 0) x <- x %>% filter(!(as.character(.data$SubjectID) %in% ids))
}

req <- c("SubjectID", "WWR", "Complexity", "ExperienceGroup")
miss <- setdiff(req, names(x))
if (length(miss) > 0) stop(paste("Missing required columns:", paste(miss, collapse = ", ")))

x <- x %>% mutate(
  SubjectID = as.factor(.data$SubjectID),
  WWR = as.factor(.data$WWR),
  Complexity = as.factor(.data$Complexity),
  ExperienceGroup = as.factor(.data$ExperienceGroup)
)

all_dvs <- c(sprintf("S%d", 1:5), sprintf("B%d", 1:3), sprintf("IPQ%d", 1:6), "IPQ_mean")
dvs <- intersect(all_dvs, names(x))

status_rows <- list()
desc_rows <- list()
type3_rows <- list()
fixed_rows <- list()
random_rows <- list()
fit_rows <- list()
emm_rows <- list()
pair_rows <- list()
warning_rows <- list()

for (dv in dvs) {
  dat <- x %>% filter(!is.na(.data[[dv]]))
  n_subjects <- dplyr::n_distinct(dat$SubjectID)
  n_rows <- nrow(dat)

  if (n_rows == 0 || n_subjects < 3) {
    status_rows[[length(status_rows) + 1]] <- data.frame(DV = dv, Status = "skipped_insufficient_data", n_rows = n_rows, n_subjects = n_subjects)
    next
  }

  desc_rows[[length(desc_rows) + 1]] <- dat %>%
    group_by(.data$WWR, .data$Complexity, .data$ExperienceGroup) %>%
    summarise(DV = dv, n = dplyr::n(), mean = mean(.data[[dv]], na.rm = TRUE), sd = sd(.data[[dv]], na.rm = TRUE), .groups = "drop") %>%
    relocate(.data$DV)

  if (dv %in% c("B1", "B2", "B3")) {
    rhs <- paste("WWR * ExperienceGroup")
  } else {
    rhs <- paste("WWR * Complexity * ExperienceGroup")
  }
  fit_obj <- try(fit_with_fallback(formula_txt = paste0(dv, " ~ ", rhs), dat = dat), silent = TRUE)
  if (inherits(fit_obj, "try-error")) {
    status_rows[[length(status_rows) + 1]] <- data.frame(DV = dv, Status = "fit_failed", Reason = as.character(fit_obj), n_rows = n_rows, n_subjects = n_subjects, caution_flag = TRUE, caution_reason = "fit_failed")
    next
  }

  fit <- fit_obj$fit
  full_formula <- fit_obj$formula
  used_re <- fit_obj$re
  fit_method <- fit_obj$method
  warning_flags <- collect_warning_flags(fit_obj$warnings)
  singular_flag <- is_singular_safe(fit)
  caution_flag <- isTRUE(warning_flags$has_convergence_warning) || isTRUE(warning_flags$has_singularity_warning) || isTRUE(singular_flag)
  caution_reason <- paste(unique(c(
    if (isTRUE(warning_flags$has_convergence_warning)) "convergence_warning" else NULL,
    if (isTRUE(warning_flags$has_singularity_warning)) "singularity_warning" else NULL,
    if (isTRUE(singular_flag)) "isSingular_TRUE" else NULL,
    if (isTRUE(warning_flags$has_kroger_warning)) "kenward_roger_fallback_warning" else NULL
  )), collapse = ";")

  status_rows[[length(status_rows) + 1]] <- data.frame(
    DV = dv,
    Status = "ok",
    Formula = full_formula,
    requested_random = "(1 + Complexity | SubjectID)",
    used_random = used_re,
    fit_method = fit_method,
    n_rows = n_rows,
    n_subjects = n_subjects,
    singular_flag = singular_flag,
    has_warning = warning_flags$has_warning,
    warning_text = warning_flags$warning_text,
    caution_flag = caution_flag,
    caution_reason = ifelse(nzchar(caution_reason), caution_reason, NA_character_)
  )

  if (warning_flags$has_warning) {
    warn_classes <- unique(c(
      if (isTRUE(warning_flags$has_convergence_warning)) "convergence_warning" else NULL,
      if (isTRUE(warning_flags$has_singularity_warning)) "singularity_warning" else NULL,
      if (isTRUE(warning_flags$has_kroger_warning)) "kenward_roger_fallback_warning" else NULL
    ))
    if (length(warn_classes) == 0) warn_classes <- "other_warning"
    for (wc in warn_classes) {
      warning_rows[[length(warning_rows) + 1]] <- data.frame(DV = dv, Source = "model_fit", warning_class = wc, warning_text = warning_flags$warning_text)
    }
  }
  if (isTRUE(singular_flag)) {
    warning_rows[[length(warning_rows) + 1]] <- data.frame(DV = dv, Source = "model_fit", warning_class = "isSingular_TRUE", warning_text = "lme4::isSingular returned TRUE")
  }

  type3_tmp <- try(extract_type3(fit, dv, full_formula, opt$`df-method`, "(1 + Complexity | SubjectID)", used_re, fit_method, n_rows, n_subjects), silent = TRUE)
  if (!inherits(type3_tmp, "try-error")) {
    type3_rows[[length(type3_rows) + 1]] <- type3_tmp
    sig_effects <- type3_tmp %>% filter(!is.na(.data$p) & .data$p < 0.05) %>% pull(.data$Effect) %>% as.character()
    emm_out <- try(make_emmeans_outputs(fit, dv, sig_effects, opt$`p-adjust`), silent = TRUE)
    if (!inherits(emm_out, "try-error")) {
      if (nrow(emm_out$emmeans) > 0) emm_rows[[length(emm_rows) + 1]] <- emm_out$emmeans
      if (nrow(emm_out$pairwise) > 0) pair_rows[[length(pair_rows) + 1]] <- emm_out$pairwise
      if (nrow(emm_out$warnings) > 0) {
        ew <- emm_out$warnings %>% rename(warning_text = .data$Warning) %>% mutate(warning_class = "emmeans_warning")
        warning_rows[[length(warning_rows) + 1]] <- ew[, c("DV", "Source", "warning_class", "warning_text")]
      }
    }
  }

  fixed_tmp <- try(extract_fixed_effects(fit, dv, opt$`df-method`), silent = TRUE)
  if (!inherits(fixed_tmp, "try-error")) fixed_rows[[length(fixed_rows) + 1]] <- fixed_tmp

  random_tmp <- try(extract_random_effects(fit, dv), silent = TRUE)
  if (!inherits(random_tmp, "try-error") && nrow(random_tmp) > 0) random_rows[[length(random_rows) + 1]] <- random_tmp

  fit_rows[[length(fit_rows) + 1]] <- data.frame(
    DV = dv,
    Formula = full_formula,
    requested_random = "(1 + Complexity | SubjectID)",
    used_random = used_re,
    fit_method = fit_method,
    n_rows = n_rows,
    n_subjects = n_subjects,
    AIC = AIC(fit),
    BIC = BIC(fit),
    logLik = as.numeric(logLik(fit)),
    minus2LL = -2 * as.numeric(logLik(fit))
  )
}

status_df <- bind_rows(status_rows)
desc_df <- bind_rows(desc_rows)
type3_df <- bind_rows(type3_rows)
fixed_df <- bind_rows(fixed_rows)
random_df <- bind_rows(random_rows)
fit_df <- bind_rows(fit_rows)
emm_df <- bind_rows(emm_rows)
pair_df <- bind_rows(pair_rows)
warning_df <- bind_rows(warning_rows)

fdr_df <- type3_df
if (nrow(fdr_df) > 0) {
  keep_mask <- !is.na(fdr_df$p)
  fdr_df$p_fdr <- NA_real_
  if (sum(keep_mask) > 0) {
    fdr_df$p_fdr[keep_mask] <- p.adjust(fdr_df$p[keep_mask], method = "fdr")
  }
  fdr_df$Sig_FDR <- vapply(fdr_df$p_fdr, sig_tier, character(1))
}

summary_df <- status_df %>%
  left_join(fit_df %>% select("DV", "AIC", "BIC", "logLik", "minus2LL", "used_random"), by = "DV")

write_csv(status_df, file.path(out_dir, "csv", "item_level_lmm_model_status.csv"))
write_csv(desc_df, file.path(out_dir, "csv", "item_level_lmm_descriptives.csv"))
write_csv(type3_df, file.path(out_dir, "csv", "item_level_lmm_type3_fixed_effects.csv"))
write_csv(fdr_df, file.path(out_dir, "csv", "item_level_lmm_type3_fixed_effects_fdr.csv"))
write_csv(fixed_df, file.path(out_dir, "csv", "item_level_lmm_fixed_effect_estimates.csv"))
write_csv(random_df, file.path(out_dir, "csv", "item_level_lmm_random_effects.csv"))
write_csv(fit_df, file.path(out_dir, "csv", "item_level_lmm_fit_indices.csv"))
write_csv(emm_df, file.path(out_dir, "csv", "item_level_lmm_emmeans.csv"))
write_csv(pair_df, file.path(out_dir, "csv", "item_level_lmm_pairwise.csv"))
write_csv(warning_df, file.path(out_dir, "csv", "item_level_lmm_warnings.csv"))

write_report_zh(file.path(out_dir, "md", "item_level_lmm_report_zh.md"), summary_df, fdr_df, pair_df, warning_df)

payload <- list(
  task = "item_level_lmm",
  dv_count = length(dvs),
  outputs = c(
    "csv/item_level_lmm_model_status.csv",
    "csv/item_level_lmm_descriptives.csv",
    "csv/item_level_lmm_type3_fixed_effects.csv",
    "csv/item_level_lmm_type3_fixed_effects_fdr.csv",
    "csv/item_level_lmm_fixed_effect_estimates.csv",
    "csv/item_level_lmm_random_effects.csv",
    "csv/item_level_lmm_fit_indices.csv",
    "csv/item_level_lmm_emmeans.csv",
    "csv/item_level_lmm_pairwise.csv",
    "csv/item_level_lmm_warnings.csv",
    "md/item_level_lmm_report_zh.md"
  ),
  modeling_note = "Each DV is fit with the same fixed-effect structure: WWR * Complexity * ExperienceGroup.",
  p_adjust = opt$`p-adjust`,
  df_method = opt$`df-method`
)
writeLines(toJSON(payload, auto_unbox = TRUE, pretty = TRUE), file.path(out_dir, "json", "item_level_lmm_summary.json"))
cat(toJSON(payload, auto_unbox = TRUE))
