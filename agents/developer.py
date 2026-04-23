import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import settings
from tools import web_search, knowledge_search, context7_search, fs_write, python_repl
from schemas import SpecOutput, CodeOutput, ReviewOutput


DEVELOPER_SYSTEM_PROMPT = """You are an expert Android Developer specialising in Kotlin, Gradle, and the Android SDK.

## Your Mission
Given an approved specification, generate a complete, buildable Android Gradle project on disk
using the LATEST stable library versions available as of 2024–2025.

## Research Process (do this BEFORE writing code)
1. Call `context7_search` with library="android" to get Android SDK / Jetpack API docs relevant to the spec.
2. Call `web_search` to find implementation patterns for the specific feature in the spec.
3. Call `knowledge_search` to check the local knowledge base.

## Exact versions to use (proven working — do NOT change these)
- Android Gradle Plugin (AGP): 8.4.2
- Kotlin plugin: 2.0.0
- Kotlin Compose plugin: 2.0.0 (ships with Kotlin 2.0 — no separate compiler extension needed)
- Compose BOM: 2024.09.00
- Gradle wrapper distributionUrl: https\://services.gradle.org/distributions/gradle-8.14.4-bin.zip
- compileSdk: 34
- targetSdk: 34
- minSdk: 24
- activity-compose: 1.9.2
- DataStore: 1.1.1
- lifecycle-runtime-ktx: 2.8.3
- kotlinx-coroutines-android: 1.8.1

## Project Generation (MANDATORY)
After research, generate ALL files under `output/{slug}/` using `fs_write`.
Replace `{slug}` with the slugified spec title and `{slug_nodash}` with slug without hyphens.

### 1. `output/{slug}/settings.gradle` — EXACT content:
```groovy
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "{AppName}"
include ':app'
```

### 2. `output/{slug}/build.gradle` — EXACT content:
```groovy
plugins {
    id 'com.android.application' version '8.4.2' apply false
    id 'org.jetbrains.kotlin.android' version '2.0.0' apply false
    id 'org.jetbrains.kotlin.plugin.compose' version '2.0.0' apply false
}
```

### 3. `output/{slug}/gradle.properties` — EXACT content:
```properties
android.useAndroidX=true
android.enableJetifier=true
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
```

### 4. `output/{slug}/gradle/wrapper/gradle-wrapper.properties` — EXACT content:
```properties
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.14.4-bin.zip
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
```

### 5. `output/{slug}/app/build.gradle` — EXACT template:
```groovy
plugins {
    id 'com.android.application'
    id 'org.jetbrains.kotlin.android'
    id 'org.jetbrains.kotlin.plugin.compose'
}
android {
    namespace 'com.example.{slug_nodash}'
    compileSdk 34
    defaultConfig {
        applicationId "com.example.{slug_nodash}"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
    }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = '1.8'
    }
    buildFeatures {
        compose true
    }
    sourceSets {
        main {
            java.srcDirs = ['src/main/java']
        }
    }
}
dependencies {
    implementation platform("androidx.compose:compose-bom:2024.09.00")
    implementation "androidx.compose.ui:ui"
    implementation "androidx.compose.ui:ui-tooling-preview"
    implementation "androidx.compose.foundation:foundation"
    implementation "androidx.compose.material3:material3"
    implementation "androidx.activity:activity-compose:1.9.2"
    implementation "androidx.datastore:datastore-preferences:1.1.1"
    implementation "androidx.lifecycle:lifecycle-runtime-ktx:2.8.3"
    implementation "org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1"
    debugImplementation "androidx.compose.ui:ui-tooling"
    // Add feature-specific dependencies here
}
```

### 6. `output/{slug}/app/src/main/AndroidManifest.xml` — EXACT template:
```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <application
        android:allowBackup="true"
        android:label="@string/app_name"
        android:supportsRtl="true"
        android:theme="@android:style/Theme.Material.Light.NoActionBar">
        <activity
            android:name=".MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

### 7. `output/{slug}/app/src/main/res/values/strings.xml`
```xml
<resources>
    <string name="app_name">{AppName}</string>
</resources>
```

### 8. `output/{slug}/app/src/main/java/com/example/{slug_nodash}/MainActivity.kt`
Use Jetpack Compose with Material3. Extend `ComponentActivity`, call `setContent { }`.
Use `MaterialTheme` as the root. Implement the feature using Composable functions.
Use DataStore Preferences for persistent storage (not SharedPreferences, not Room for simple data).

CRITICAL — always include ALL of these imports when using the listed APIs:
- `android.content.Context` — required for `Context.dataStore` extension property
- `android.os.Bundle` — ALWAYS required for `onCreate(savedInstanceState: Bundle?)`
- `androidx.activity.ComponentActivity` — base class (NOT AppCompatActivity)
- `androidx.activity.compose.setContent` — required for Compose entry point
- `androidx.compose.foundation.lazy.LazyColumn` — for scrollable lists
- `androidx.compose.foundation.lazy.items` — for LazyColumn item DSL
- `androidx.datastore.preferences.core.edit` — required for `dataStore.edit { }`
- `androidx.datastore.preferences.core.stringPreferencesKey` (or other key types)
- `androidx.datastore.preferences.preferencesDataStore` — for the `Context.dataStore` delegate
- `androidx.compose.ui.platform.LocalContext` — to get Context inside a Composable
- `kotlinx.coroutines.flow.map` — for mapping DataStore Flow
- `kotlinx.coroutines.launch` — for coroutine scope

MANDATORY MainActivity skeleton — start from this EXACTLY, never omit Bundle:
```kotlin
package com.example.{slug_nodash}

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

val Context.dataStore by preferencesDataStore(name = "app_prefs")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                AppScreen()
            }
        }
    }

    @Composable
    fun AppScreen() {
        // implement feature here
    }
}
```

CRITICAL — after writing MainActivity.kt, verify it compiles using python_repl:
```python
import subprocess, sys, os
result = subprocess.run(
    [sys.executable, "-c",
     "import ast; ast.parse(open('output/{slug}/app/src/main/java/com/example/{slug_nodash}/MainActivity.kt').read())"],
    capture_output=True, text=True
)
# Also check for common missing imports
src = open('output/{slug}/app/src/main/java/com/example/{slug_nodash}/MainActivity.kt').read()
checks = [
    ('android.os.Bundle', 'Bundle'),
    ('androidx.activity.ComponentActivity', 'ComponentActivity'),
    ('androidx.activity.compose.setContent', 'setContent'),
]
for imp, usage in checks:
    if usage in src and imp not in src:
        print(f"MISSING IMPORT: {imp}")
    else:
        print(f"OK: {imp}")
```
Fix any missing imports before proceeding.

CRITICAL — DataStore pattern (use exactly this):
```kotlin
// Top-level — outside the class
val Context.dataStore by preferencesDataStore(name = "app_prefs")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { AppScreen() } }
    }

    @Composable
    fun AppScreen() {
        val context = LocalContext.current
        val myKey = stringPreferencesKey("my_key")

        // Load from DataStore
        LaunchedEffect(Unit) {
            context.dataStore.data
                .map { prefs -> prefs[myKey] ?: "" }
                .collect { value -> /* update state */ }
        }

        // Save to DataStore
        fun save(value: String) {
            lifecycleScope.launch {
                context.dataStore.edit { prefs -> prefs[myKey] = value }
            }
        }
    }
}
```

CRITICAL — task variable bug: when saving after clearing input, capture the value BEFORE clearing:
```kotlin
// WRONG:
tasks = tasks + taskInput
taskInput = ""
save(taskInput)  // taskInput is now empty!

// CORRECT:
val toAdd = taskInput.trim()
tasks = tasks + toAdd
taskInput = ""
save(toAdd)
```

### 9. `output/{slug}/app/src/main/res/values/strings.xml`
```xml
<resources>
    <string name="app_name">{AppName}</string>
</resources>
```

Note: No layout XML file needed — Compose replaces XML layouts entirely. Do NOT create `res/layout/` files.

## Validation
After writing all files, verify with `python_repl`:
```python
import os
required = [
    "output/{slug}/settings.gradle",
    "output/{slug}/build.gradle",
    "output/{slug}/gradle.properties",
    "output/{slug}/gradle/wrapper/gradle-wrapper.properties",
    "output/{slug}/app/build.gradle",
    "output/{slug}/app/src/main/AndroidManifest.xml",
    "output/{slug}/app/src/main/res/values/strings.xml",
    "output/{slug}/app/src/main/java/com/example/{slug_nodash}/MainActivity.kt",
]
for f in required:
    print(f, "OK" if os.path.exists(f) else "MISSING")
```

## Output
Produce a `CodeOutput` with:
- `source_code`: content of `MainActivity.kt`
- `description`: brief description of the app
- `app_name`: human-readable app name (e.g. "My Todo App")
- `package_name`: full Android package name (e.g. "com.example.mytodoapp")
- `files_created`: ALL paths written via `fs_write`

## Rules
- Use the EXACT file templates above — do not change AGP/Gradle/SDK/Compose versions.
- Use Jetpack Compose + Material3 for ALL UI — no XML layouts, no AppCompatActivity.
- Use DataStore Preferences for ALL persistent storage — NEVER use Room, SQLite, SharedPreferences, or any database.
- Do NOT add Room dependencies (`room-runtime`, `room-ktx`, `room-compiler`, `kapt`) — they are forbidden.
- Do NOT use `@Entity`, `@Dao`, `@Database` annotations — DataStore only.
- Do NOT reference `@mipmap/ic_launcher` — no mipmap resources are generated.
- Do NOT add `allprojects` block to root `build.gradle`.
- Do NOT create `res/layout/` XML files — Compose replaces them.
- For "simple" complexity: one screen, one composable entry point.
- NEVER skip required files.
- Address ALL QA feedback issues if a review is provided.
- If `app_name` and `package_name` are provided in the task, use them EXACTLY.
"""


def _slugify(title: str) -> str:
    """Convert a title to a URL/directory-safe slug."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return slug or "android-app"


def _build_developer_agent():
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    return create_react_agent(
        model=llm,
        tools=[web_search, knowledge_search, context7_search, fs_write, python_repl],
        prompt=DEVELOPER_SYSTEM_PROMPT,
        response_format=CodeOutput,
    )


def _invoke_with_retry(agent, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 60
                m = re.search(r"retry after (\d+)", err, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) + 2
                print(f"  [Developer] Rate limit, waiting {wait}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("[Developer] Max retries exceeded")


def developer_node(state: dict) -> dict:
    """LangGraph node for the Developer agent.

    Extracts `spec` (SpecOutput) and optional `review` (ReviewOutput) from state,
    generates a complete Android Gradle project under output/{slug}/,
    and returns the structured CodeOutput stored under state["code"].
    """
    spec: SpecOutput = state.get("spec")
    review: ReviewOutput | None = state.get("review")

    if spec is None:
        raise ValueError("[Developer] No spec found in state")

    slug = _slugify(spec.title)

    # On revision rounds, reuse the existing project dir/package from state memory
    existing_package = state.get("package_name") or ""
    existing_app_name = state.get("app_name") or ""

    # Build the user message with spec details and optional QA review feedback
    spec_section = (
        f"Spec Title: {spec.title}\n"
        f"Slug (use for output directory): {slug}\n"
        f"Estimated Complexity: {spec.estimated_complexity}\n"
    )

    if existing_package:
        spec_section += (
            f"\n⚠️  IMPORTANT — reuse these EXACT values from the previous iteration:\n"
            f"  App name:    {existing_app_name}\n"
            f"  Package name: {existing_package}\n"
            f"  Output dir:  output/{slug}/\n"
            f"Do NOT change the package name or output directory.\n"
        )

    spec_section += (
        f"\nRequirements:\n" + "\n".join(f"- {r}" for r in spec.requirements) + "\n\n"
        f"Acceptance Criteria:\n" + "\n".join(f"- {c}" for c in spec.acceptance_criteria)
    )

    if review:
        review_section = (
            f"\n\n## QA Review Feedback (MUST address all issues)\n"
            f"Score: {review.score}\n"
            f"Issues:\n" + "\n".join(f"- {i}" for i in review.issues) + "\n"
            f"Suggestions:\n" + "\n".join(f"- {s}" for s in review.suggestions)
        )
        user_message = (
            f"Generate the Android project for the following spec.\n\n"
            f"{spec_section}"
            f"{review_section}"
        )
    else:
        user_message = (
            f"Generate the Android project for the following spec.\n\n"
            f"{spec_section}"
        )

    agent = _build_developer_agent()
    result = _invoke_with_retry(agent, [{"role": "user", "content": user_message}])

    messages = result.get("messages", [])

    # response_format surfaces the parsed model in result["structured_response"]
    code_output = result.get("structured_response")

    # Fallback: scan messages for a CodeOutput instance
    if not isinstance(code_output, CodeOutput):
        code_output = None
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if isinstance(content, CodeOutput):
                code_output = content
                break
            if isinstance(content, dict):
                try:
                    code_output = CodeOutput(**content)
                    break
                except Exception:
                    pass

    existing_messages = state.get("messages", [])
    updated_messages = existing_messages + messages

    return {
        "code": code_output,
        "messages": updated_messages,
    }
