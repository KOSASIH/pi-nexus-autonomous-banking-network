# mainnet_migration/migration_views.py

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import MigrationUser, MigrationTransaction, MigrationContract
from .forms import MigrationForm
from .utils import send_migration_transaction

@login_required
def migration_view(request):
    user = request.user
    migration_user = MigrationUser.objects.get(user=user)
    if migration_user.migration_status == 'pending':
        if request.method == 'POST':
            form = MigrationForm(request.POST)
            if form.is_valid():
                mainnet_address = form.cleaned_data['mainnet_address']
                testnet_address = form.cleaned_data['testnet_address']
                migration_user.mainnet_address = mainnet_address
                migration_user.testnet_address = testnet_address
                migration_user.migration_status = 'in_progress'
                migration_user.save()
                send_migration_transaction(migration_user)
                return redirect('migration_status')
        else:
            form = MigrationForm()
        return render(request, 'migration_form.html', {'form': form})
    elif migration_user.migration_status == 'in_progress':
        return render(request, 'migration_in_progress.html')
    elif migration_user.migration_status == 'completed':
        return render(request, 'migration_completed.html')

@login_required
def migration_status_view(request):
    user = request.user
    migration_user = MigrationUser.objects.get(user=user)
    transactions = MigrationTransaction.objects.filter(user=migration_user)
    return render(request, 'migration_status.html', {'transactions': transactions})
