// profileCustomization.js

class ProfileCustomization {
    constructor(userId) {
        this.userId = userId;
        this.profile = {
            displayName: '',
            profilePicture: '',
            bio: '',
        };
    }

    // Get current profile information
    getProfile() {
        return this.profile;
    }

    // Update profile information
    updateProfile(newProfileData) {
        this.profile = { ...this.profile, ...newProfileData };
        console.log(`Profile updated for user ${this.userId}:`, this.profile);
    }

    // Reset profile to default values
    resetProfile() {
        this.profile = {
            displayName: '',
            profilePicture: '',
            bio: '',
        };
        console.log(`Profile reset to default for user ${this.userId}.`);
    }
}

// Example usage
const userProfile = new ProfileCustomization('user123');
userProfile.updateProfile({
    displayName: 'John Doe',
    profilePicture: 'https://example.com/profile.jpg',
    bio: 'Finance enthusiast and mentor.',
});

console.log('Current Profile:', userProfile.getProfile());

userProfile.resetProfile();
console.log('Profile after reset:', userProfile.getProfile());
