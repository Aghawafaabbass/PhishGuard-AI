import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import urllib.parse
import tldextract
import re
from urllib.parse import urlparse

# ────────────────────────────────────────────────
# Feature extraction helpers (same as before)

def count_dots(url): return url.count('.')
def count_subdomain_level(extracted):
    subdomain = extracted.subdomain
    return len(subdomain.split('.')) + 1 if subdomain else 0

def count_path_level(path):
    if path in ['', '/']: return 0
    return len([p for p in path.split('/') if p])

def url_length(url): return len(url)
def count_dashes(url): return url.count('-')
def count_dashes_hostname(hostname): return hostname.count('-')
def has_at_symbol(url): return int('@' in url)
def has_tilde(url): return int('~' in url)
def count_underscore(url): return url.count('_')
def count_percent(url): return url.count('%')
def count_query_components(query): return len(query.split('&')) if query else 0
def count_ampersand(url): return url.count('&')
def count_hash(url): return url.count('#')
def count_numeric_chars(url): return sum(c.isdigit() for c in url)
def no_https(scheme): return 1 if scheme.lower() != 'https' else 0

def has_random_string(path):
    if not path: return 0
    alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', path)
    if len(alphanumeric) > 12 and sum(c.isalpha() for c in alphanumeric) < len(alphanumeric)*0.4:
        return 1
    return 0

def has_ip_address(hostname):
    parts = hostname.split('.')
    if len(parts) != 4: return 0
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)

def domain_in_subdomains(subdomain, domain):
    return int(domain in subdomain) if subdomain else 0

def domain_in_paths(path, domain):
    return int(domain.lower() in path.lower())

def https_in_hostname(hostname): return int('https' in hostname.lower())
def hostname_length(hostname): return len(hostname)
def path_length(path): return len(path)
def query_length(query): return len(query)
def double_slash_in_path(path): return int('//' in path)
def count_sensitive_words(url):
    sensitive = ['login','signin','bank','secure','account','update','verify','password','paypal','ebay']
    return sum(url.lower().count(word) for word in sensitive)

def has_embedded_brand_name(url):
    common = ['paypal','amazon','google','apple','microsoft']
    return int(any(b in url.lower() for b in common))

def extract_features_from_url(url):
    try:
        parsed = urlparse(url)
        extracted = tldextract.extract(url)
        scheme, hostname, path, query = parsed.scheme, (parsed.hostname or ''), (parsed.path or ''), (parsed.query or '')
        full_url = url.strip()

        features = {
            'NumDots': count_dots(full_url),
            'SubdomainLevel': count_subdomain_level(extracted),
            'PathLevel': count_path_level(path),
            'UrlLength': url_length(full_url),
            'NumDash': count_dashes(full_url),
            'NumDashInHostname': count_dashes_hostname(hostname),
            'AtSymbol': has_at_symbol(full_url),
            'TildeSymbol': has_tilde(full_url),
            'NumUnderscore': count_underscore(full_url),
            'NumPercent': count_percent(full_url),
            'NumQueryComponents': count_query_components(query),
            'NumAmpersand': count_ampersand(full_url),
            'NumHash': count_hash(full_url),
            'NumNumericChars': count_numeric_chars(full_url),
            'NoHttps': no_https(scheme),
            'RandomString': has_random_string(path),
            'IpAddress': has_ip_address(hostname),
            'DomainInSubdomains': domain_in_subdomains(extracted.subdomain, extracted.domain),
            'DomainInPaths': domain_in_paths(path, extracted.domain),
            'HttpsInHostname': https_in_hostname(hostname),
            'HostnameLength': hostname_length(hostname),
            'PathLength': path_length(path),
            'QueryLength': query_length(query),
            'DoubleSlashInPath': double_slash_in_path(path),
            'NumSensitiveWords': count_sensitive_words(full_url),
            'EmbeddedBrandName': has_embedded_brand_name(full_url),
            # Page-dependent features set to 0
            'PctExtHyperlinks': 0.0, 'PctExtResourceUrls': 0.0, 'ExtFavicon': 0,
            'InsecureForms': 0, 'RelativeFormAction': 0, 'ExtFormAction': 0,
            'AbnormalFormAction': 0, 'PctNullSelfRedirectHyperlinks': 0.0,
            'FrequentDomainNameMismatch': 0, 'FakeLinkInStatusBar': 0,
            'RightClickDisabled': 0, 'PopUpWindow': 0, 'SubmitInfoToEmail': 0,
            'IframeOrFrame': 0, 'MissingTitle': 0, 'ImagesOnlyInForm': 0,
            # RT approximations
            'SubdomainLevelRT': 1 if count_subdomain_level(extracted) > 2 else -1,
            'UrlLengthRT': -1 if len(full_url) > 75 else 1,
            'PctExtResourceUrlsRT': 0, 'AbnormalExtFormActionR': 0,
            'ExtMetaScriptLinkRT': 0, 'PctExtNullSelfRedirectHyperlinksRT': 0,
        }

        feature_order = [
            'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
            'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
            'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
            'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
            'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
            'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
            'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
            'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms', 'RelativeFormAction',
            'ExtFormAction', 'AbnormalFormAction', 'PctNullSelfRedirectHyperlinks',
            'FrequentDomainNameMismatch', 'FakeLinkInStatusBar', 'RightClickDisabled',
            'PopUpWindow', 'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
            'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
            'PctExtResourceUrlsRT', 'AbnormalExtFormActionR', 'ExtMetaScriptLinkRT',
            'PctExtNullSelfRedirectHyperlinksRT'
        ]

        return pd.DataFrame([features])[feature_order]
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        cols = feature_order
        return pd.DataFrame(np.zeros((1, len(cols))), columns=cols)

# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model('phishguard_xgb.json')
    return model

model = load_model()

def predict_url(url):
    if not url.strip() or not url.lower().startswith(('http://', 'https://')):
        return None, "Please enter a valid URL (http/https)", "orange"
    features = extract_features_from_url(url)
    proba = model.predict_proba(features)[0][1]
    label = 1 if proba >= 0.5 else 0
    verdict = "🚨 High risk – Phishing" if label == 1 else "✅ Looks safe – Legitimate"
    color = "red" if label == 1 else "green"
    return proba, verdict, color

# ────────────────────────────────────────────────
# UI
st.set_page_config(page_title="PhishGuard AI", layout="wide")
st.title("🛡️ PhishGuard AI – Phishing URL Detector")
st.markdown("Paste a URL or upload CSV with 'url' column for batch check.")

tab1, tab2 = st.tabs(["Single URL", "Batch CSV"])

with tab1:
    url_input = st.text_input("URL to check:", "https://www.google.com")
    if st.button("Analyze URL", type="primary"):
        with st.spinner("Checking..."):
            proba, verdict, color = predict_url(url_input)
            if proba is not None:
                st.metric("Phishing Probability", f"{proba:.1%}")
                st.markdown(f"<h3 style='color:{color};'>{verdict}</h3>", unsafe_allow_html=True)
                if proba > 0.8: st.error("Very high confidence phishing – avoid!")
                elif proba > 0.5: st.warning("Suspicious – be careful.")
            else:
                st.warning(verdict)

with tab2:
    uploaded = st.file_uploader("Upload CSV (needs 'url' column)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'url' not in df.columns:
            st.error("CSV must have a column called 'url'")
        else:
            results = []
            for idx, row in df.iterrows():
                u = str(row['url'])
                proba, verdict, _ = predict_url(u)
                results.append({'url': u, 'probability': proba, 'verdict': verdict})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.format({'probability': '{:.1%}'}), use_container_width=True)
            phish_count = (res_df['probability'] >= 0.5).sum()
            st.metric("Detected as Phishing", phish_count)

st.caption("PhishGuard AI • XGBoost + lexical features • Educational demo")
