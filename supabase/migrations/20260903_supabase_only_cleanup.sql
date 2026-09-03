-- Phase A/B: Supabase-only runtime persistence.
-- Apply in Supabase SQL Editor (or the project's migration workflow); this
-- file is not executed by the runtime application.

ALTER TABLE speaker_profiles
    DROP CONSTRAINT IF EXISTS speaker_profiles_user_id_fkey;
ALTER TABLE speaker_profiles
    ADD CONSTRAINT speaker_profiles_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE enrollment_samples
    DROP CONSTRAINT IF EXISTS enrollment_samples_user_id_fkey;
ALTER TABLE enrollment_samples
    ADD CONSTRAINT enrollment_samples_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE tasks
    DROP CONSTRAINT IF EXISTS tasks_user_id_fkey;
ALTER TABLE tasks
    ADD CONSTRAINT tasks_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE schedules
    DROP CONSTRAINT IF EXISTS schedules_user_id_fkey;
ALTER TABLE schedules
    ADD CONSTRAINT schedules_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

ALTER TABLE private_notes
    DROP CONSTRAINT IF EXISTS private_notes_user_id_fkey;
ALTER TABLE private_notes
    ADD CONSTRAINT private_notes_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- Preserve audit evidence after a user is deleted while removing the former
-- user association. user_id is nullable in the existing schema.
ALTER TABLE audit_logs
    DROP CONSTRAINT IF EXISTS audit_logs_user_id_fkey;
ALTER TABLE audit_logs
    ADD CONSTRAINT audit_logs_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL;

-- Raw enrollment audio is not retained. Keep the column for existing schema
-- compatibility, but make metadata-only enrollment rows valid.
ALTER TABLE enrollment_samples
    ALTER COLUMN audio_path DROP NOT NULL;
