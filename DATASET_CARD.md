# VoiceCodeBench Dataset Card

This card documents VoiceCodeBench using datasheet-style and data-statement
principles. It is intended to complement the concise README and the benchmark
paper.

## Motivation

VoiceCodeBench evaluates a narrow production failure mode in ASR: whether a
transcript preserves exact structured values that downstream software must parse,
route, store, compare, or execute.

WER remains useful for broad transcript quality, but it can obscure value-level
failures. VoiceCodeBench therefore reports entity-sensitive metrics for exact
canonical-value recovery in addition to WER.

## Dataset Summary

- Name: VoiceCodeBench.
- Language: English.
- Modality: Human-recorded speech audio with text metadata.
- Task: Raw-audio ASR evaluation for exact structured-token recovery.
- Split: Test-only.
- Size: 300 recordings, 5.587 hours of WAV audio.
- Speakers: 85 anonymized speaker IDs.
- Entities: 1,482 target entities.
- Entity types: 26.
- Workflow domains: 8.
- Baseline outputs: 13 ASR systems.

## Composition

Each row in `data/metadata.jsonl` contains:

- `file_name` and `audio_id`.
- `language`, `duration`, `domain`, `scenario`, and `difficulty`.
- Coarse anonymized speaker metadata.
- Audio-quality metadata.
- Three transcript layers: `template`, `acoustic`, and `canonical`.
- Target entities with `id`, `type`, `role`, `acoustic`, and `canonical`.

The metadata row format is schema version 1. Rows use three transcript layers:

- `template`: script text with entity placeholders.
- `acoustic`: the words the speaker is expected to say aloud.
- `canonical`: the exact written value expected by downstream software.

The 26 entity types are:

```text
email_address, phone_number, phone_extension, person_or_team_name,
postal_address, url, ip_address, port_number, command, cli_flag, file_path,
environment_variable, code_symbol, version, reference_id, product_code,
account_or_record_number, currency_amount, percentage, measurement,
plain_number, date, time, acronym_or_initialism, spelled_sequence, domain_term
```

Entity taxonomy groups:

| Group | Entity types |
| --- | --- |
| `contact_and_routing` | `email_address`, `phone_number`, `phone_extension`, `person_or_team_name`, `postal_address` |
| `network_and_web` | `url`, `ip_address`, `port_number` |
| `code_and_system` | `command`, `cli_flag`, `file_path`, `environment_variable`, `code_symbol`, `version` |
| `identifiers` | `reference_id`, `product_code`, `account_or_record_number` |
| `numeric_and_measurement` | `currency_amount`, `percentage`, `measurement`, `plain_number`, `date`, `time` |
| `language_form` | `acronym_or_initialism`, `spelled_sequence`, `domain_term` |

Workflow-domain distribution:

| Domain | Recordings |
| --- | ---: |
| `contact_routing` | 45 |
| `technical_it_dev` | 55 |
| `retail_logistics_order` | 45 |
| `finance_billing` | 40 |
| `healthcare_admin` | 35 |
| `legal_insurance_government` | 35 |
| `education_workplace` | 25 |
| `dense_mixed_stress` | 20 |

Difficulty distribution:

| Difficulty | Recordings | Entity target | Word target | Description |
| --- | ---: | --- | --- | --- |
| `light` | 30 | 2-3 | 90-135 | Low entity load with common formats and clear context. |
| `standard` | 114 | 4 | 100-170 | Moderate entity load with common formats and clear context. |
| `dense` | 93 | 5-6 | 120-180 | Higher entity load with mixed alphanumeric tokens, punctuation-sensitive entities, uncommon names, or multiple similar entities. |
| `stress` | 63 | 7-8 | 150-210 | Short segments with cross-type interference, similar IDs, or 7-8 compact target entities. |

Domain descriptions:

| Domain | Description | Scenarios |
| --- | --- | --- |
| `contact_routing` | Routing requests, confirming contact details, escalating support cases, and leaving callback instructions with synthetic emails, phone numbers, extensions, names, departments, and spelled components. | `callback_instructions`, `support_escalation`, `department_routing`, `mailing_update`, `case_handoff`, `after_hours_contact` |
| `technical_it_dev` | Debugging, deployment, account setup, network configuration, incident response, and developer support with IPs, URLs, ports, versions, CLI flags, filenames, environment variables, acronyms, and technical terms. | `deployment_debugging`, `incident_response`, `network_configuration`, `developer_support`, `account_setup`, `release_validation`, `database_migration` |
| `retail_logistics_order` | Purchase orders, returns, warehouse exceptions, warranty claims, subscriptions, and shipping issues with SKUs, serial numbers, tracking IDs, model numbers, quantities, product names, emails, and phone numbers. | `order_status`, `return_authorization`, `warehouse_exception`, `warranty_claim`, `inventory_update`, `subscription_shipment` |
| `finance_billing` | Invoices, reconciliations, payment disputes, budgeting, investment explanations, payroll, taxes, and account updates with currency, percentages, invoice IDs, account numbers, dates, amounts, acronyms, and finance terms. | `invoice_reconciliation`, `payment_dispute`, `budget_update`, `payroll_adjustment`, `tax_preparation`, `investment_account_update` |
| `healthcare_admin` | Appointments, referrals, lab follow-ups, pharmacy calls, insurance preauthorization, patient portal support, and care instructions with measurements, dosages, dates, appointment IDs, acronyms, spelled names, phone numbers, and emails. | `appointment_scheduling`, `referral_follow_up`, `lab_results`, `pharmacy_call`, `insurance_preauth`, `patient_portal_support` |
| `legal_insurance_government` | Case intake, claims, policy updates, court filings, compliance checks, public records, and benefits administration with case IDs, policy numbers, exhibit labels, statute references, addresses, dates, acronyms, and formal terms. | `case_intake`, `claim_update`, `policy_change`, `court_filing`, `compliance_check`, `benefits_administration` |
| `education_workplace` | School administration, course registration, HR updates, facilities requests, training records, meeting logistics, and internal helpdesk tickets with course codes, room numbers, employee IDs, emails, spelled names, acronyms, and dates. | `course_registration`, `hr_update`, `facilities_request`, `training_record`, `meeting_logistics`, `internal_helpdesk` |
| `dense_mixed_stress` | Compact mixed-workflow stress tests with several structured tokens in one plausible segment, targeting 7-8 mostly compact entities per datapoint. | `incident_bridge_summary`, `multi_order_support_note`, `clinic_billing_call`, `legal_case_update`, `finance_it_cutover`, `warehouse_claim_review` |

The distribution is controlled for diagnostic coverage. It is not intended to
estimate how often these workflows or entity types occur in production speech.

## Collection Process

Scenarios and structured values are synthetic. The dataset was constructed
entity-first: workflow domain, difficulty, entity count, and entity types were
specified before complete scripts were written. Each target entity has an
acoustic form and a canonical written form.

Transcript and entity generation was LLM-assisted and repository-constrained.
Candidate items were checked for metadata validity, domain fit, naturalness,
uniqueness, entity consistency, and recoverability.

Audio was recorded by human speakers reading the acoustic transcript layer.
Speakers were instructed to read naturally but clearly while preserving dictated
punctuation phrases, spelling sequences, casing cues, and formatting
instructions. Accepted recordings are released without additional
post-processing.

## Contributor Consent And Compensation

Recordings were collected remotely through a crowdsourcing platform from paid
contributors who consented to dataset use and release. The public metadata
contains anonymous speaker IDs and coarse speaker fields only. Individual
contributor identities, contact information, and payment records are not
released.

## Speaker Metadata

Speaker metadata is coarse and anonymized. It should be treated as descriptive,
not as a validated demographic ground truth.

Sex labels:

| Label | Recordings |
| --- | ---: |
| `female` | 152 |
| `male` | 144 |
| `non_binary` | 4 |

Age buckets:

| Bucket | Recordings |
| --- | ---: |
| `18_25` | 76 |
| `26_35` | 53 |
| `36_45` | 102 |
| `46_55` | 54 |
| `56_65` | 10 |
| `66_plus` | 5 |

Accent labels are self-reported or platform-provided coarse categories in the
metadata. They are included for transparency and should not be used for
profiling or broad claims about demographic performance.

## Privacy And Sensitive Content

The released audio may contain identifiable voice characteristics. The dataset
should not be used for speaker identification, biometric modeling, voice
cloning, demographic profiling, or attempts to re-identify contributors.

Scripts and structured values are synthetic. Contact details, account numbers,
credentials, live operational systems, and private records are avoided by
construction. Documentation domains, fictional phone-number ranges, private or
reserved IP ranges, and synthetic addresses/identifiers are used where
appropriate.

## Intended Use

VoiceCodeBench is intended for:

- Comparing ASR systems on exact structured-token recovery.
- Tracking ASR regressions over time.
- Studying the gap between transcript readability and downstream value
  correctness.
- Identifying fragile entity types before deploying ASR-backed workflows.
- Evaluating whether additional safeguards are needed for high-risk entity
  classes.

## Prohibited Or Out-Of-Scope Use

VoiceCodeBench is not intended for:

- Training, fine-tuning, or post-training speech models.
- Speaker identification, biometric modeling, voice cloning, or demographic
  profiling.
- Estimating natural production frequencies of workflow domains or entity types.
- Measuring all-purpose ASR quality across languages, acoustic environments, or
  conversational settings.
- Hidden-leaderboard evaluation where public labels are treated as secret.

Reports using VoiceCodeBench should disclose model version, evaluation date,
inference settings, and any prompting, custom vocabulary, post-processing,
fine-tuning, or canonicalization.

## Metrics And Scoring

VoiceCodeBench reports:

- Format-invariant WER after lowercasing and word tokenization. Each target
  entity span accepts its acoustic or canonical rendering as an alternative
  reference.
- CTEM: the fraction of target entities whose canonical value is recoverable
  from the ASR transcript.
- TSR: the fraction of recordings for which every target entity is recovered.
- Per-entity-type recovery rates.

This alternative-reference scoring prevents standard written forms such as
`212-555-0100` from being counted as recognition errors against spelled-out
acoustic forms. Words outside target entity spans use ordinary edit distance.

Entity scoring uses an LLM-assisted recoverability verifier for the tracked
baseline outputs. The verifier prompt and response schema are versioned in
`scripts/voice_code_bench/verifiers/openai_gpt_5_5_v1.json`, and verifier
decisions are stored with evidence and reasons in `baselines/predictions/*.json`.
A stratified audit sample is released in `audit/verifier_audit_samples.csv`.

## Limitations

VoiceCodeBench is English-only and focused on compact workplace-style dictation.
It does not cover meetings, overlapping speech, casual conversation, broadcast
audio, voice search, noisy field recordings, telephony codecs, or long-form
dictation.

The content is synthetic, so it may not capture every distributional property of
real support calls, developer conversations, clinical documentation, logistics
workflows, or financial operations.

Exact structured-token scoring requires policy choices. The current baseline
release uses an LLM-assisted verifier, which is more flexible than literal
string matching but introduces dependence on a verifier model and prompt.

Commercial ASR systems change over time. Baseline rankings are tied to the
evaluation dates and provider behavior recorded in the baseline artifacts.

## Maintenance And Versioning

The public release is versioned by repository commits and Hugging Face dataset
revisions. Metadata schema version, entity taxonomy, difficulty bands, and
domain descriptions are documented in this card. Baseline artifacts include run
metadata such as provider, model, endpoint or API, evaluation date, mode, and
inference settings.

Future releases should preserve prior revisions and document changes to audio,
metadata, entity taxonomy, scoring policy, verifier versions, and baseline
systems. Issues or pull requests should include enough information to reproduce
the affected scoring or metadata behavior.

## Licensing

VoiceCodeBench is released under the MIT License in `LICENSE`. Users are
responsible for using the human voice audio consistently with the dataset's
intended-use and prohibited-use guidance.
